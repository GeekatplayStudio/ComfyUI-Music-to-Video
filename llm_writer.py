"""
Local-LLM storyboard writer for ComfyUI-Music-to-Video.

Geekatplay Studio - Vladimir Chopine

The keyword tables in nodes.py can only say what someone already wrote down:
24 imagery rules, 12 locations, 16 props. A song about grief, chess or a
harvest matches none of them and falls through to a generic default, and the
cinematography is picked by `index % len(table)`, so card 1 and card 15 are
framed identically. This module hands the same job to a small instruct model
that has actually read the song.

Two stages, because a 4B model cannot hold eighty prompts in one JSON object
without drifting or truncating:

  A. one call  - read the whole song: who it is about, where it happens,
                 what recurs, what the palette is. Small output.
  B. k calls   - card + motion prompt per segment, in batches, each batch given
                 stage A as context so the deck stays coherent across the song.

Every failure path raises LLMUnavailable. A missing model, an out-of-memory
card or a reply that will not parse must never kill a queue - the caller drops
back to the rule tables instead. The fallback is whole-deck on purpose: mixing
model prose and table prose in one storyboard reads worse than either alone.
"""

import gc
import json
import os
import re
import sys

import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None


class LLMUnavailable(RuntimeError):
    """The local model could not be loaded, or would not produce usable JSON."""


def _safe_print(text):
    """Console-safe print - lyrics carry characters cp1252 cannot encode."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(str(text).encode(encoding, errors="replace").decode(encoding, errors="replace"))


# Reference kinds the render nodes know how to shoot. REFERENCE_VIEWS in
# nodes.py only has sheets for these three; anything else has no view list.
REF_KINDS = ("character", "scene", "prop")

# Segments per stage B call. Eight keeps the reply short enough that a 4B model
# finishes the JSON, while still giving it neighbouring lyrics for continuity.
BATCH_SIZE = 8


def resolve_model_path(name):
    """
    Accept either a folder under ComfyUI/models/LLM or a Hugging Face repo id.

    A local folder wins, so an air-gapped install never silently reaches for
    the network when the user meant the copy already on disk.
    """
    name = str(name or "").strip()
    if not name:
        raise LLMUnavailable("No LLM model name given.")

    if os.path.isdir(name):
        return name

    if folder_paths is not None:
        root = os.path.join(folder_paths.models_dir, "LLM")
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            return candidate

    # Not on disk - treat it as a repo id and let transformers resolve it from
    # the HF cache (or download it, if the user has network).
    return name


class LocalWriter:
    """
    Loads an instruct model, answers a handful of chat turns, unloads.

    Used as a context manager so the weights are gone before Part 2 loads the
    image model. The prompt stage runs before any renderer in every shipped
    workflow, so it has the card to itself - but only if it gives it back.
    """

    def __init__(self, model_name, max_new_tokens=1600, seed=0, temperature=0.85):
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __enter__(self):
        self._load()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.unload()
        return False

    def _load(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise LLMUnavailable(
                "transformers is not installed. It normally ships with ComfyUI; "
                "install it with:  python -m pip install transformers accelerate"
            )

        path = resolve_model_path(self.model_name)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        _safe_print(f"[LLMWriter] Loading '{path}' on {self.device} ({dtype})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        except Exception as exc:
            raise LLMUnavailable(f"Could not load the tokenizer for '{path}': {exc}")

        if self.tokenizer.chat_template is None:
            raise LLMUnavailable(
                f"'{path}' has no chat template, so it is a base model rather than an "
                "instruct model. Point llm_model at an -Instruct / -it checkpoint."
            )

        self.model = self._load_weights(path, dtype)
        self.model.eval()
        _safe_print(f"[LLMWriter] Ready: {path}")

    def _load_weights(self, path, dtype):
        """
        Try a text-only causal LM first, then a vision-language one.

        Users already have Gemma 3 and Qwen-VL checkpoints on disk for LTX and
        for captioning; those load through a different auto class but generate
        text perfectly well, so accepting them saves a redundant download.
        """
        from transformers import AutoModelForCausalLM

        auto_classes = [AutoModelForCausalLM]
        try:
            from transformers import AutoModelForImageTextToText
            auto_classes.append(AutoModelForImageTextToText)
        except ImportError:
            pass

        # "auto" rather than "cuda": accelerate places what fits on the card and
        # spills the rest to system RAM. Pinning to cuda would OOM on an 8-12GB
        # card - which is most of the user base - and drop them back to the rule
        # tables permanently, with only a log line to say why.
        device_map = "auto" if self.device == "cuda" else self.device

        errors = []
        for auto in auto_classes:
            for dtype_kwarg in ("dtype", "torch_dtype"):   # renamed in transformers 5
                try:
                    return auto.from_pretrained(
                        path,
                        device_map=device_map,
                        trust_remote_code=False,
                        **{dtype_kwarg: dtype},
                    )
                except TypeError as exc:
                    if dtype_kwarg == "dtype" and "dtype" in str(exc):
                        continue        # old transformers - retry as torch_dtype
                    errors.append(f"{auto.__name__}: {exc}")
                    break
                except Exception as exc:
                    errors.append(f"{auto.__name__}: {type(exc).__name__}: {exc}")
                    break

        raise LLMUnavailable(
            f"Could not load '{path}' as an instruct model.\n  " + "\n  ".join(errors)
        )

    def unload(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def chat(self, system, user, max_new_tokens=None, seed_offset=0):
        """One system+user turn in, the assistant's text out."""
        if self.model is None:
            raise LLMUnavailable("chat() called before the model was loaded.")

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # A few templates reject a system role; fold it into the user turn.
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": system + "\n\n" + user}],
                tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        torch.manual_seed(self.seed + seed_offset)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens or self.max_new_tokens),
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


# --------------------------------------------------------------------------
# Reading JSON back out of a small model's reply.

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text):
    """
    Pull the first JSON object or array out of a reply.

    Small models wrap JSON in fences, introduce it with a sentence, or append
    an explanation afterwards. All three are recoverable and none of them are
    worth failing a four-minute song over.
    """
    if text is None:
        raise ValueError("empty reply")

    candidates = []
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    for chunk in candidates:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Whichever bracket opens first wins. Trying "{" first would read a
        # stage B array of shot objects as only its first object, quietly
        # throwing away the rest of the batch.
        pairs = sorted((("{", "}"), ("[", "]")),
                       key=lambda p: chunk.find(p[0]) if p[0] in chunk else len(chunk) + 1)
        for opener, closer in pairs:
            start = chunk.find(opener)
            end = chunk.rfind(closer)
            if start == -1 or end <= start:
                continue
            blob = chunk[start:end + 1]
            for attempt in (blob, re.sub(r",(\s*[}\]])", r"\1", blob)):
                try:
                    return json.loads(attempt)
                except json.JSONDecodeError:
                    continue

    raise ValueError("no JSON object found in the reply")


def _clean(value, limit=900):
    """One-line, length-capped string - prompts must not carry newlines."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].strip()


# --------------------------------------------------------------------------
# Stage A - read the whole song once.

_STAGE_A_SYSTEM = """You are a music video director planning a shoot.

You will be given a song: its lyrics laid out on a timeline, and a measured
analysis of how it sounds. Decide what the video is about and reply with a
single JSON object, nothing else.

Required shape:
{
  "subject": "the recurring person or figure the video follows, described so an
              artist could draw the same one twice: age, build, face, hair,
              wardrobe, colours. One sentence.",
  "setting": "the world every shot lives in: place, era, time of day, weather.
              One sentence.",
  "palette": "the colour and light treatment for the whole video. One phrase.",
  "mood": "one or two words",
  "recurring_imagery": ["3 to 5 concrete visual motifs drawn from the lyrics"],
  "references": [
    {"id": "snake_case_id",
     "kind": "character" | "scene" | "prop",
     "name": "short human-readable name",
     "prompt": "a standalone visual description of this exact thing, detailed
                enough that two different renders match: shape, colour,
                material, wear, distinguishing marks"}
  ]
}

Rules for references:
- Exactly one "character" reference, and it must describe the subject above.
- Add a "scene" reference for each distinct location the song actually visits
  (at most 3), and a "prop" reference for each object the lyrics return to and
  that must look identical every time (at most 3).
- Never write a camera angle, view or shot type into a reference prompt. The
  angles are added later; a reference that names one will be rendered wrong.
- Take everything from this song. Do not invent a stock music-video character.
"""


def _approach_note(approach):
    """The user's chosen prompt approach, as a directive both stages obey."""
    a = str(approach or "")
    if a.startswith("Abstract"):
        return ("APPROACH: Abstract. Every card is a non-representational image - "
                "colour fields, bold shapes, gesture and motion energy standing in "
                "for the music. No people, no faces, no recognisable places or "
                "objects; a lyric may only surface as a half-formed echo. The "
                "character reference then describes a recurring FORM or colour "
                "motif, not a person.")
    if a.startswith("Lyrics"):
        return ("APPROACH: Lyrics-focused. Each shot literally shows what that "
                "segment's own words are about - stay close to the sung line, "
                "and do not import imagery from other verses.")
    if a.startswith("Concert"):
        return ("APPROACH: Concert performance. The entire video is the singer "
                "performing this song live on ONE stage. The character reference "
                "is the singer: face, haircut and one stage outfit, kept "
                "absolutely identical in every card. The scene references are "
                "views of that single concert stage (wide from the crowd, front "
                "at the microphone, from the wings) - the location NEVER changes. "
                "Every card shows the singer mid-performance - singing into the "
                "microphone, working the crowd, hitting the chorus - matched to "
                "that segment's lyric and energy; only the lighting rig, the "
                "framing and the crowd's energy move with the song.")
    if a.startswith("Creative"):
        return ("APPROACH: Creative. Read the whole song as one story; let mood, "
                "symbolism and imagery leak between verses, favour symbolic and "
                "surreal touches over literal illustration of the line.")
    return ""


def _song_brief(data, max_segments=60):
    """The song as the model sees it: timeline, words, and what it sounds like."""
    segments = data.get("segments") or []
    music = data.get("music") or {}
    lines = []

    if music:
        lines.append(
            "SOUND: {tempo} BPM, {key} {mode}, energy {energy}, "
            "brightness {bright} Hz, mood '{emo}', suggested look '{look}'.".format(
                tempo=music.get("tempo", "?"), key=music.get("key", "?"),
                mode=music.get("mode", "?"), energy=music.get("energy", "?"),
                bright=music.get("brightness_hz", "?"),
                emo=str(music.get("emotional_tone", "")).replace("_", " "),
                look=str(music.get("look", "")).replace("_", " ")))
        lines.append("")

    lines.append(f"TIMELINE: {len(segments)} segments.")
    for seg in segments[:max_segments]:
        lyric = seg.get("lyrics") or "(instrumental)"
        lines.append(
            "  [{idx:03d}] {start:6.2f}-{end:6.2f}s  {section:<18} {lyric}".format(
                idx=seg.get("segment_index", 0),
                start=float(seg.get("start_time", 0.0)),
                end=float(seg.get("end_time", 0.0)),
                section=str(seg.get("section_label", "") or "-")[:18],
                lyric=_clean(lyric, 160)))
    if len(segments) > max_segments:
        lines.append(f"  ... {len(segments) - max_segments} more segments")
    return "\n".join(lines)


def _validate_stage_a(obj):
    if not isinstance(obj, dict):
        raise ValueError("stage A did not return an object")

    subject = _clean(obj.get("subject"), 400)
    setting = _clean(obj.get("setting"), 400)
    if not subject or not setting:
        raise ValueError("stage A returned no subject or no setting")

    references = []
    seen = set()
    for raw in (obj.get("references") or []):
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "")).strip().lower()
        if kind not in REF_KINDS:
            continue
        ref_id = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("id", "")).strip().lower()).strip("_")
        if not ref_id or ref_id in seen:
            continue
        prompt = _clean(raw.get("prompt"), 700)
        if not prompt:
            continue
        seen.add(ref_id)
        references.append({
            "id": ref_id,
            "kind": kind,
            "name": _clean(raw.get("name"), 80) or ref_id.replace("_", " "),
            "prompt": prompt,
        })

    # The card deck always leans on a character reference; without one the
    # per-card assignment has nothing to fall back to.
    if not any(r["kind"] == "character" for r in references):
        references.insert(0, {"id": "character", "kind": "character",
                              "name": "main character", "prompt": subject})

    return {
        "subject": subject,
        "setting": setting,
        "palette": _clean(obj.get("palette"), 300),
        "mood": _clean(obj.get("mood"), 80),
        "recurring_imagery": [_clean(x, 120) for x in (obj.get("recurring_imagery") or [])
                              if _clean(x, 120)][:5],
        "references": references,
    }


# --------------------------------------------------------------------------
# Stage B - one card and one motion prompt per segment.

_STAGE_B_SYSTEM = """You are writing shot prompts for a music video that has
already been planned. You will be given the plan, then a batch of segments.

For each segment write two prompts and reply with a single JSON array, nothing
else. One array entry per segment, in the order given:

[
  {"segment_index": <the number you were given>,
   "card": "a still image prompt for the keyframe that opens this segment.
            Name the shot size and angle, what the subject is doing, where they
            are, and the light. Vary the framing from the segment before -
            never two identical shot sizes in a row.",
   "motion": "how that still moves over the next few seconds: subject motion
              first, then camera motion. Present tense, concrete, one shot, no
              cuts.",
   "reference_id": "the id of the reference this shot must match"}
]

Rules:
- Write what is visible. No lyrics quoted, no song titles, no text in frame.
- The subject must be recognisably the same person in every card.
- Draw the imagery from this segment's own words. If a segment is instrumental,
  carry the mood and the recurring motifs instead.
- The motion prompt describes ONE continuous shot. Never write a cut, an edit,
  a transition or a montage.
- reference_id must be one of the ids listed in the plan.
"""


def _plan_summary(plan):
    ref_lines = "\n".join(
        f'  - id "{r["id"]}" ({r["kind"]}): {r["name"]} - {r["prompt"][:180]}'
        for r in plan["references"])
    imagery = ", ".join(plan["recurring_imagery"]) or "(none)"
    return (f"PLAN\n"
            f"  subject : {plan['subject']}\n"
            f"  setting : {plan['setting']}\n"
            f"  palette : {plan['palette']}\n"
            f"  mood    : {plan['mood']}\n"
            f"  motifs  : {imagery}\n"
            f"REFERENCES\n{ref_lines}")


def _batch_brief(segments):
    lines = ["SEGMENTS"]
    for seg in segments:
        lines.append(
            "  segment_index {idx}: {start:.2f}-{end:.2f}s ({dur:.1f}s), {section}, "
            "energy {energy}\n    lyrics: {lyric}".format(
                idx=seg.get("segment_index", 0),
                start=float(seg.get("start_time", 0.0)),
                end=float(seg.get("end_time", 0.0)),
                dur=float(seg.get("duration", 0.0)),
                section=str(seg.get("section_label", "") or "-"),
                energy=seg.get("local_energy", "-"),
                lyric=_clean(seg.get("lyrics") or "(instrumental)", 240)))
    return "\n".join(lines)


def _validate_stage_b(obj, expected_indices, valid_ref_ids):
    """Map the reply back onto segment indices, keeping only usable entries."""
    if isinstance(obj, dict):
        obj = obj.get("segments") or obj.get("shots") or [obj]
    if not isinstance(obj, list):
        raise ValueError("stage B did not return an array")

    out = {}
    for position, raw in enumerate(obj):
        if not isinstance(raw, dict):
            continue
        card = _clean(raw.get("card"))
        motion = _clean(raw.get("motion"))
        if not card or not motion:
            continue

        # Trust the stated index when it is one we asked for; small models
        # sometimes renumber from zero, so fall back to arrival order.
        try:
            idx = int(raw.get("segment_index"))
        except (TypeError, ValueError):
            idx = None
        if idx not in expected_indices:
            idx = expected_indices[position] if position < len(expected_indices) else None
        if idx is None or idx in out:
            continue

        ref_id = str(raw.get("reference_id", "")).strip().lower()
        if ref_id not in valid_ref_ids:
            ref_id = None
        out[idx] = {"card": card, "motion": motion, "reference_id": ref_id}

    missing = [i for i in expected_indices if i not in out]
    if missing:
        raise ValueError(f"no usable entry for segment(s) {missing}")
    return out


# --------------------------------------------------------------------------

def write_storyboard(storyboard_data, model_name, seed=0, temperature=0.85,
                     max_new_tokens=1600, batch_size=BATCH_SIZE, approach=""):
    """
    Plan a whole music video with a local instruct model.

    Returns (card_prompts, animation_prompts, reference_plan, plan) with the
    same shapes the rule engine produces, so the render nodes downstream do not
    know or care which writer produced them.
    """
    data = json.loads(storyboard_data) if isinstance(storyboard_data, str) else dict(storyboard_data)
    segments = data.get("segments") or []
    if not segments:
        raise LLMUnavailable("The timeline has no segments to write prompts for.")
    card_count = int(data.get("card_count") or (len(segments) + 1))

    with LocalWriter(model_name, max_new_tokens=max_new_tokens, seed=seed,
                     temperature=temperature) as writer:

        # --- stage A -------------------------------------------------------
        brief = _song_brief(data)
        note = _approach_note(approach)
        if note:
            brief = note + "\n\n" + brief
        plan = None
        for attempt in range(2):
            reply = writer.chat(_STAGE_A_SYSTEM, brief, max_new_tokens=1200,
                                seed_offset=attempt)
            try:
                plan = _validate_stage_a(extract_json(reply))
                break
            except Exception as exc:
                _safe_print(f"[LLMWriter] Stage A attempt {attempt + 1} unusable ({exc}).")
        if plan is None:
            raise LLMUnavailable("The model never returned a usable song plan.")

        _safe_print(f"[LLMWriter] Subject : {plan['subject'][:110]}")
        _safe_print(f"[LLMWriter] Setting : {plan['setting'][:110]}")
        _safe_print(f"[LLMWriter] {len(plan['references'])} reference(s): "
                    + ", ".join(f"{r['id']}({r['kind']})" for r in plan["references"]))

        # --- stage B -------------------------------------------------------
        plan_text = _plan_summary(plan)
        if note:
            plan_text = note + "\n\n" + plan_text
        valid_ref_ids = {r["id"] for r in plan["references"]}
        shots = {}
        batch_size = max(1, int(batch_size))

        # The deck is one card longer than the timeline: the extra card is the
        # frame the final segment animates into. Ask for it as its own shot -
        # reusing the last segment's card renders the same image twice.
        work = list(segments)
        if card_count > len(segments):
            finale = dict(segments[-1])
            finale["segment_index"] = len(segments)
            finale["lyrics"] = (_clean(segments[-1].get("lyrics"), 200)
                                + "  >> THIS IS THE CLOSING FRAME the video ends on. "
                                "Resolve the story; do not repeat the previous shot.")
            finale["section_label"] = "Closing frame"
            work.append(finale)

        for start in range(0, len(work), batch_size):
            batch = work[start:start + batch_size]
            indices = [int(s.get("segment_index", start + n)) for n, s in enumerate(batch)]
            user = plan_text + "\n\n" + _batch_brief(batch)

            got = None
            for attempt in range(2):
                reply = writer.chat(_STAGE_B_SYSTEM, user,
                                    max_new_tokens=260 * len(batch) + 200,
                                    seed_offset=1000 + start + attempt)
                try:
                    got = _validate_stage_b(extract_json(reply), indices, valid_ref_ids)
                    break
                except Exception as exc:
                    _safe_print(f"[LLMWriter] Segments {indices[0]}-{indices[-1]} "
                                f"attempt {attempt + 1} unusable ({exc}).")
            if got is None:
                raise LLMUnavailable(
                    f"The model could not write segments {indices[0]}-{indices[-1]}. "
                    "Try a larger model, or switch prompt_writer back to the rule tables.")
            shots.update(got)
            _safe_print(f"[LLMWriter] Wrote segments {indices[0]}-{indices[-1]} "
                        f"({len(shots)}/{len(work)}).")

    # --- assemble the same outputs the rule engine returns ------------------
    character_id = next((r["id"] for r in plan["references"] if r["kind"] == "character"),
                        plan["references"][0]["id"])
    card_prompts, animation_prompts, assignment = [], [], {}

    for i in range(card_count):
        source = shots.get(i, {})
        card = source.get("card") or ""
        if not card:
            raise LLMUnavailable(f"No card prompt was produced for card {i}.")
        card_prompts.append(card)
        assignment[str(i)] = source.get("reference_id") or character_id

    for seg in segments:
        idx = int(seg.get("segment_index", 0))
        animation_prompts.append(shots[idx]["motion"])

    used = set(assignment.values())
    references = [r for r in plan["references"] if r["id"] in used]
    return (card_prompts,
            animation_prompts,
            {"references": references, "assignment": assignment},
            plan)
