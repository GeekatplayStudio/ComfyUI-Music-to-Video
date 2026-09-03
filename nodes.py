"""
ComfyUI-Music-to-Video
Music Video Storyboard & Video Generator for ComfyUI.

Geekatplay Studio - Vladimir Chopine
https://github.com/GeekatplayStudio/ComfyUI-Music-to-Video

Thank you for your support!
  * Star the project on GitHub - it genuinely helps.
  * Subscribe on YouTube:
      https://www.youtube.com/@geekatplay        (English)
      https://www.youtube.com/@geekatplay-ru     (Russian)
      https://www.youtube.com/@v-code-studio     (new channel - V-Code Studio)
"""

import os
import re
import sys
import time
import random
import zlib
import json
import subprocess
import shutil
import gc
import torch
import numpy as np

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import comfy.samplers
except ImportError:
    comfy = None

try:
    import comfy.model_management as mm
except ImportError:
    mm = None

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:
    try:
        from comfy_execution.graph import ExecutionBlocker
    except ImportError:
        ExecutionBlocker = None

# VIDEO objects know how to write themselves; SegmentVideoSaver uses this to put
# each clip in the run folder under its own segment number.
try:
    from comfy_api.latest import Types as ComfyVideoTypes
except ImportError:
    ComfyVideoTypes = None


# --------------------------------------------------------------------------
# Geekatplay Studio - Vladimir Chopine
GAP_AUTHOR = "Geekatplay Studio - Vladimir Chopine"
GAP_GITHUB = "https://github.com/GeekatplayStudio/ComfyUI-Music-to-Video"
GAP_YOUTUBE = ("https://www.youtube.com/@geekatplay",
               "https://www.youtube.com/@geekatplay-ru",
               "https://www.youtube.com/@v-code-studio")


def _gap_credit(width=78):
    return "\n".join([
        "-" * width,
        f" {GAP_AUTHOR}",
        " Thank you for your support! Please star the project on GitHub",
        f"   {GAP_GITHUB}",
        " and subscribe on YouTube:",
        f"   {GAP_YOUTUBE[0]}   (English)",
        f"   {GAP_YOUTUBE[1]}   (Russian)",
        f"   {GAP_YOUTUBE[2]}   (V-Code Studio - new channel)",
        "-" * width,
    ])


class AnyType(str):
    """Wildcard socket type: compares equal to every other type."""
    def __ne__(self, other):
        return False


ANY = AnyType("*")

# A video segment must stay inside the LTX 2.5 clip window.
MIN_SEGMENT_SEC = 5.0
MAX_SEGMENT_SEC = 10.0

# How the motion prompts read the song - see the "Prompt approaches" section.
PROMPT_APPROACHES = [
    "Creative (artistic music video - mood, symbolism, ideas leak between verses)",
    "Lyrics-focused (closely follow what is being sung)",
    "Abstract (non-literal - pure mood, colour and motion)",
    "Concert (performance video - the singer live on one stage, identical outfit & haircut)",
]

SUBJECT_MODES = [
    "Auto (detect from music theme & lyrics)",
    "Singer / Performer",
    "Single Person / Main Character",
    "Band / Group (Multiple People)",
    "Crowd / Atmospheric People",
    "No People (Scenery & Objects only)",
]

# Who writes the card and motion prompts. The rule tables below can only say
# what someone already wrote into them, and they frame by index, so a song
# whose subject nobody anticipated gets generic cards. A local instruct model
# reads the actual song instead - at the cost of a download and a load.
PROMPT_WRITERS = [
    "rule tables (offline, no download)",
    "local LLM (reads the song, needs a model)",
]

# Instruct checkpoints live here, alongside every other model class. Registered
# so folder_paths.models_dir resolves 'models/LLM/<name>' for local folders.
if folder_paths is not None:
    try:
        os.makedirs(os.path.join(folder_paths.models_dir, "LLM"), exist_ok=True)
    except OSError:
        pass



def _safe_print(text):
    """
    Console-safe print. Transcribed lyrics can contain characters the Windows
    console codepage (cp1252) cannot encode, and a raw print() would raise
    UnicodeEncodeError and fail the node.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(str(text).encode(encoding, errors="replace").decode(encoding, errors="replace"))


# Shared "run this step again" behaviour. Returning NaN makes the cache signature
# never match, so the node re-executes on every queue while the toggle is on.
FORCE_RERUN_INPUT = ("BOOLEAN", {
    "default": False,
    "label_on": "RERUN this step",
    "label_off": "use cached result",
})


class ForceRerunMixin:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("force_rerun"):
            return float("nan")
        return "cached"


def _block(reason=None):
    """Stop downstream execution. Returns None-safe blocker for older ComfyUI builds."""
    if ExecutionBlocker is None:
        return None
    return ExecutionBlocker(reason)


AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus', '.wma', '.aiff', '.aif')


def _resolve_audio_path(raw):
    """
    Turn whatever the user typed/pasted into a real file path.
    Accepts absolute paths, ~ shortcuts, %VARS%, quoted paths (Windows 'Copy as path'
    wraps in quotes), and bare filenames living in ComfyUI/input.
    """
    if not raw or not str(raw).strip():
        return ""

    candidate = str(raw).strip().strip('"').strip("'")
    candidate = os.path.expandvars(os.path.expanduser(candidate))

    if os.path.isfile(candidate):
        return os.path.abspath(candidate)

    # Bare filename: look in ComfyUI/input, the way the old dropdown used to.
    if folder_paths is not None and not os.path.isabs(candidate):
        in_input = os.path.join(folder_paths.get_input_directory(), candidate)
        if os.path.isfile(in_input):
            return os.path.abspath(in_input)

    if os.path.isdir(candidate):
        raise ValueError(f"'{candidate}' is a folder, not a song file. Point 'audio_path' at the file itself.")

    raise ValueError(
        f"Song file not found: '{candidate}'. Paste the full path to the file "
        f"(supported: {', '.join(AUDIO_EXTENSIONS)}), or connect a 'Load Audio' node instead."
    )


def _probe_duration(path):
    """Read exact duration with ffprobe. Raises rather than guessing - a wrong
    duration would silently mis-time the whole music video."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except FileNotFoundError:
        raise ValueError(
            "ffprobe was not found on PATH, so the song length cannot be measured. "
            "Install ffmpeg (https://ffmpeg.org/download.html), or connect a 'Load Audio' "
            "node to the 'audio' input - that path needs no ffprobe."
        )

    if res.returncode != 0 or not res.stdout.strip():
        raise ValueError(f"ffprobe could not read '{path}'. Is it a valid audio file?\n{res.stderr.strip()}")
    return float(res.stdout.strip())


def _duration_from_waveform(audio):
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError("Connected audio reports an invalid sample rate.")
    return float(waveform.shape[-1]) / sample_rate


def _write_temp_wav(audio, basename="videoclipmaker_song.wav"):
    """
    Dump the connected AUDIO tensor to a WAV so ffmpeg can mux the real song into the
    final video. Uses the stdlib wave module - no extra dependency.
    """
    import wave

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim == 3:          # [batch, channels, samples]
        waveform = waveform[0]
    if waveform.ndim == 1:          # [samples]
        waveform = waveform.unsqueeze(0)

    samples = waveform.transpose(0, 1).contiguous().float().clamp(-1.0, 1.0).cpu().numpy()
    pcm = (samples * 32767.0).astype(np.int16)

    out_dir = folder_paths.get_temp_directory() if folder_paths else os.path.abspath(".")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, basename)

    with wave.open(path, "wb") as handle:
        handle.setnchannels(int(pcm.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

    print(f"[AudioLyricsSegmenter] Wrote song track for muxing -> {path}")
    return path


INSTRUMENTAL = "(instrumental)"

# --------------------------------------------------------------------------
# Music analysis. The look of a video should follow the sound: tempo, key,
# energy and timbre decide whether this is romantic warm film or hard neon.
# These are signal heuristics, not a trained genre classifier - the numbers are
# reported so you can see exactly why a look was chosen, and override it.

# Krumhansl-Schmuckler key profiles, used to guess major vs minor.
_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# (label, style preset to use, palette + treatment added to every prompt)
MUSIC_LOOKS = {
    "dark_electronic": (
        "Cyberpunk",
        "electric teal and magenta neon against deep black, hard specular highlights, "
        "wet reflective surfaces, high contrast, cold artificial light"),
    "driving_rock": (
        "Cinematic film",
        "gritty high-contrast grade, crushed blacks, dust and haze in hard backlight, "
        "desaturated with deep red accents"),
    "upbeat_pop": (
        "Cinematic film",
        "vivid saturated colour, bright bouncy key light, clean glossy surfaces, "
        "playful primary accents"),
    "romantic_ballad": (
        "Cinematic film",
        "warm golden hour palette, soft diffused light, gentle halation and bloom, "
        "creamy shallow focus, amber and rose tones"),
    "melancholic": (
        "Cinematic film",
        "desaturated blue-grey palette, flat overcast light, muted shadows, "
        "cold empty space around the subject"),
    "acoustic_intimate": (
        "Photorealistic",
        "natural earthy palette, soft window light, warm wood and fabric texture, "
        "quiet unforced composition"),
    "ambient_dreamy": (
        "Surreal dreamlike",
        "pale washed-out palette, heavy atmospheric haze, glowing diffuse light, "
        "soft edges dissolving into light"),
    "cinematic_neutral": (
        "Cinematic film",
        "balanced cinematic grade, controlled contrast, motivated practical lighting"),
    "orchestral_classical": (
        "Cinematic film",
        "opulent painterly palette, chandelier and candlelight warmth, marble, "
        "velvet and gold leaf, deep theatrical shadow"),
    "roots_americana": (
        "Photorealistic",
        "warm sun-faded earth tones, natural golden light through dust and haze, "
        "wood grain, denim and worn leather texture, honest documentary grade"),
}

# What the world of the video LOOKS like, decided by how the song SOUNDS.
# The palette above is colour and light; this is the actual visual environment
# the genre implies - a dark electronic track happens in a cyberpunk city even
# if no lyric ever says "neon". Lyrics then adjust what happens INSIDE that
# world; on an instrumental track this description carries the video alone.
MUSIC_WORLDS = {
    "dark_electronic":
        "a cyberpunk world - dense neon signage bleeding into rain, holographic "
        "light on wet chrome and glass, brutalist megastructures, silhouetted "
        "crowds under artificial glow",
    "driving_rock":
        "a raw industrial world - concrete, steel and scaffolding, dust and haze "
        "in hard backlight, sparks, worn leather and road-scarred surfaces",
    "upbeat_pop":
        "a bright stylised pop world - saturated colour-blocked spaces, glossy "
        "clean surfaces, playful bold geometry, sunlit streets and confetti light",
    "romantic_ballad":
        "a warm romantic world - golden-hour light through curtains and trees, "
        "soft lived-in interiors, candle flames, every highlight blooming gently",
    "melancholic":
        "a muted melancholic world - overcast streets and empty rooms, rain "
        "tracing down glass, cold blue-grey distances, long quiet shadows",
    "acoustic_intimate":
        "an intimate handmade world - warm wood and fabric, window light, small "
        "rooms and porches, honest natural texture, nothing polished",
    "ambient_dreamy":
        "a weightless dream world - pale endless spaces, drifting fog, floating "
        "light, horizons dissolving into glow",
    "cinematic_neutral":
        "a grounded cinematic world - real locations, motivated practical light, "
        "controlled contrast and believable depth",
    "orchestral_classical":
        "a classical world - grand halls and colonnades, marble and velvet, "
        "chandelier light and candle flame, painterly baroque depth",
    "roots_americana":
        "a roots americana world - dirt roads and fence lines, fields at golden "
        "hour, weathered barns and porches, pickup trucks, river banks and "
        "campfire smoke, everything sun-worn and real",
}

# Wardrobe follows the sound, not a fixed persona. A city song is techwear in a
# dark-electronic track and a wool coat in a melancholic one; nothing here ever
# dresses two different songs the same way. (Fixed persona strings used to live
# in PERSONA_HINTS and put the same figure in the same coat on every run.)
LOOK_WARDROBE = {
    "dark_electronic": "in dark layered techwear, neon reflections tracing the fabric",
    "driving_rock": "in worn leather and denim, road dust on their boots",
    "upbeat_pop": "in bold colour-blocked modern clothes, crisp and bright",
    "romantic_ballad": "in soft warm-toned clothing that catches the golden light",
    "melancholic": "in muted layered clothing, collar turned up against the cold",
    "acoustic_intimate": "in simple natural fabrics, unstyled and lived-in",
    "ambient_dreamy": "in pale flowing fabric that drifts with the air",
    "cinematic_neutral": "in grounded contemporary clothing with one strong accent",
    "orchestral_classical": "in elegant formal dress, tailored and timeless",
    "roots_americana": "in worn denim, flannel and boots, sun on weathered fabric",
}


def _music_world(music):
    """One sentence of visual atmosphere derived purely from how the song sounds."""
    if not music:
        return ""
    world = MUSIC_WORLDS.get(music.get("look", ""), "")
    emo = str(music.get("emotional_tone", "")).replace("_", " ")
    if world and emo:
        return f"{world}; the whole frame feels {emo}"
    return world


def _estimate_key(chroma_mean):
    """Correlate the average chroma against major/minor profiles."""
    best = (None, None, -2.0)
    for shift in range(12):
        rotated = np.roll(chroma_mean, -shift)
        for name, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            prof = np.asarray(profile, dtype=np.float64)
            a = rotated - rotated.mean()
            b = prof - prof.mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            score = float(np.dot(a, b) / denom) if denom else 0.0
            if score > best[2]:
                best = (_NOTE_NAMES[shift], name, score)
    return best[0], best[1], best[2]


def _classify_music(tempo, mode, energy, brightness, percussive, flatness):
    """
    Turn the measurements into one of the looks above.

    Thresholds are calibrated against real mixed/mastered tracks, not theory.
    Measured across six finished songs: peak-normalised RMS 0.14-0.24,
    spectral centroid 1900-3400 Hz (0.18-0.31 of Nyquist), percussive share
    0.17-0.54, spectral flatness 0.010-0.043. Textbook-looking cutoffs like
    "percussive > 0.55" never fire on real music.
    """
    fast = tempo >= 118
    slow = tempo <= 92
    minor = (mode == "minor")
    bright = brightness >= 0.26          # spectral centroid / Nyquist
    loud = energy >= 0.175               # peak-normalised RMS
    quiet = energy < 0.125
    punchy = percussive >= 0.35          # share of energy in the percussive layer
    very_punchy = percussive >= 0.45
    noisy = flatness >= 0.028            # synthetic / distorted timbre

    # Orchestral / classical: almost no percussive layer and a very tonal
    # (low-flatness) spectrum. Real drums or synth grit disqualify it long
    # before tempo does.
    # Thresholds sit BELOW the measured floor of real produced songs
    # (percussive 0.17+, flatness 0.010+), so only genuinely un-drummed,
    # purely tonal recordings land here.
    if percussive < 0.15 and flatness < 0.010 and not noisy:
        return "orchestral_classical"
    if fast and minor and (bright or noisy) and punchy:
        return "dark_electronic"
    if minor and very_punchy and loud and not slow:
        return "driving_rock"
    if fast and not minor:
        return "upbeat_pop"
    if slow and not minor and not loud:
        return "romantic_ballad"
    if slow and minor:
        return "melancholic"
    if quiet and not punchy and not noisy:
        return "acoustic_intimate"
    if quiet and not punchy:
        return "ambient_dreamy"
    if minor and (bright or noisy) and punchy:
        return "dark_electronic"
    if not minor and punchy:
        return "upbeat_pop"
    return "cinematic_neutral"


# ---- learned genre classification -----------------------------------------
# The heuristics above only see tempo/key/timbre, so a bright fast country
# track is indistinguishable from pop. A small Audio Spectrogram Transformer
# fine-tuned on GTZAN actually names the genre; its label picks the look and
# the heuristic becomes the fallback. Runs through transformers (already
# shipped with ComfyUI); the ~350MB checkpoint auto-downloads to the HF cache
# on first use. Failure of any kind just keeps the heuristic look.

GENRE_MODEL_REPO = "dima806/music_genres_classification"

# GTZAN label -> look table key. Anything unmapped keeps the heuristic look.
GENRE_TO_LOOK = {
    "country": "roots_americana",
    "blues": "roots_americana",
    "folk": "roots_americana",
    "classical": "orchestral_classical",
    "metal": "driving_rock",
    "rock": "driving_rock",
    "disco": "upbeat_pop",
    "pop": "upbeat_pop",
    "hiphop": "dark_electronic",
    "reggae": "acoustic_intimate",
    "jazz": "cinematic_neutral",
}

# Trust the model only when it is clearly ahead of the pack; a mumbled 0.2
# top score on a genre-bending track should not override the audio heuristics.
GENRE_MIN_SCORE = 0.45


def _classify_genre_model(y, sr):
    """
    Returns (genre, score) from the GTZAN AST model, or (None, 0.0) on any
    failure. `y` is the mono waveform already loaded for the librosa analysis.
    """
    try:
        import librosa
        from transformers import pipeline
    except ImportError:
        return None, 0.0

    clf = None
    try:
        # Classify a window from the heart of the song - intros mislead.
        max_len = 60 * sr
        if y.size > max_len:
            start = (y.size - max_len) // 2
            y = y[start:start + max_len]
        y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)

        device = 0 if torch.cuda.is_available() else -1
        _safe_print(f"[MusicAnalysis] Genre model '{GENRE_MODEL_REPO}'...")
        clf = pipeline("audio-classification", model=GENRE_MODEL_REPO, device=device)
        results = clf({"raw": y16, "sampling_rate": 16000}, top_k=3)
        if not results:
            return None, 0.0
        top = results[0]
        _safe_print("[MusicAnalysis] Genre: "
                    + ", ".join(f"{r['label']} {r['score']:.2f}" for r in results))
        return str(top["label"]).lower(), float(top["score"])
    except Exception as exc:
        _safe_print(f"[MusicAnalysis] Genre model unavailable ({type(exc).__name__}: {exc}); "
                    "using signal heuristics for the look.")
        return None, 0.0
    finally:
        del clf
        _free_vram()


def _analyse_music(audio_path, max_seconds=180.0):
    """
    Returns a rich dict describing how the song sounds, emotional tone, and section structure.
    Never raises: if librosa is unavailable the pipeline just skips the styling.
    """
    try:
        import librosa
    except ImportError:
        _safe_print("[MusicAnalysis] librosa not available - skipping music-based styling.")
        return None

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=max_seconds)
        if y.size < sr:
            return None

        # librosa 0.11's beat_track(y=...) path returns 0 BPM on many tracks;
        # feeding it a precomputed onset envelope is the reliable route.
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_raw, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo = float(np.atleast_1d(tempo_raw)[0])
        beat_times = [round(float(t), 3)
                      for t in librosa.frames_to_time(beat_frames, sr=sr)]
        harmonic, percussion = librosa.effects.hpss(y)
        h_energy = float(np.mean(harmonic ** 2))
        p_energy = float(np.mean(percussion ** 2))
        percussive = p_energy / (h_energy + p_energy + 1e-12)

        peak = float(np.max(np.abs(y))) or 1.0
        norm_y = y / peak

        rms_frames = librosa.feature.rms(y=norm_y, frame_length=2048, hop_length=512)[0]
        energy = float(np.mean(rms_frames))

        centroid_hz = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        brightness = centroid_hz / (sr / 2.0)
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
        key, mode, confidence = _estimate_key(np.mean(chroma, axis=1))

        # Advanced spectral & timbral feature extraction
        try:
            contrast_db = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
        except Exception:
            contrast_db = 20.0
        try:
            rolloff_hz = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
        except Exception:
            rolloff_hz = centroid_hz * 1.8
        try:
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=norm_y)))
        except Exception:
            zcr = 0.05

        # Emotion & Affect Dimension Mapping (Arousal & Valence)
        arousal = min(1.0, max(0.0, (tempo - 60) / 100 * 0.4 + energy * 2.5 + percussive * 0.3))
        mode_bias = 0.2 if mode == "major" else -0.2
        valence = min(1.0, max(-1.0, mode_bias + (brightness - 0.2) * 1.5 + (tempo - 100) / 100 * 0.3))

        if arousal >= 0.6 and valence >= 0.2:
            emotional_tone = "triumphant_euphoric"
        elif arousal >= 0.6 and valence < -0.2:
            emotional_tone = "intense_aggressive"
        elif arousal < 0.4 and valence >= 0.0:
            emotional_tone = "peaceful_intimate"
        elif arousal < 0.4 and valence < 0.0:
            emotional_tone = "melancholic_aching"
        elif arousal >= 0.5:
            emotional_tone = "driving_kinetic"
        else:
            emotional_tone = "balanced_cinematic"

        look = _classify_music(tempo, mode, energy, brightness, percussive, flatness)
        look_source = "signal heuristics"

        genre, genre_score = _classify_genre_model(y, sr)
        if genre and genre_score >= GENRE_MIN_SCORE and GENRE_TO_LOOK.get(genre):
            look = GENRE_TO_LOOK[genre]
            look_source = f"genre model ({genre} {genre_score:.2f})"
        _safe_print(f"[MusicAnalysis] Look '{look}' chosen by {look_source}.")

        preset, palette = MUSIC_LOOKS[look]

        frame_sec = 512 / sr
        rms_profile = [round(float(v), 4) for v in rms_frames[::10]]

        return {
            "tempo": round(tempo, 1),
            "beat_times": beat_times,
            "key": key,
            "mode": mode,
            "key_confidence": round(confidence, 2),
            "energy": round(energy, 4),
            "brightness_hz": round(centroid_hz, 1),
            "brightness": round(brightness, 3),
            "percussive_ratio": round(percussive, 3),
            "flatness": round(flatness, 5),
            "contrast_db": round(contrast_db, 1),
            "rolloff_hz": round(rolloff_hz, 1),
            "zcr": round(zcr, 4),
            "arousal": round(arousal, 2),
            "valence": round(valence, 2),
            "emotional_tone": emotional_tone,
            "genre": genre or "",
            "genre_confidence": round(genre_score, 2),
            "look": look,
            "look_source": look_source,
            "suggested_preset": preset,
            "palette": palette,
            "rms_profile": rms_profile,
            "frame_sec": round(frame_sec * 10, 3),
        }
    except Exception as exc:
        _safe_print(f"[MusicAnalysis] Could not analyse the audio ({exc}); skipping styling.")
        return None


def _annotate_segments_with_music_structure(segments, music, total_duration):
    """
    Annotate each segment with its musical structure (intro, verse, build-up, chorus/climax, bridge, outro)
    and time-varying energy dynamics.
    """
    if not segments or not music:
        return

    rms_profile = music.get("rms_profile") or []
    frame_sec = float(music.get("frame_sec") or 0.23)
    avg_energy = float(music.get("energy") or 0.15)
    num_segs = len(segments)

    seg_energies = []
    for seg in segments:
        st, et = seg["start_time"], seg["end_time"]
        if rms_profile and frame_sec > 0:
            idx_start = max(0, int(st / frame_sec))
            idx_end = min(len(rms_profile), int(et / frame_sec) + 1)
            chunk = rms_profile[idx_start:idx_end]
            local_e = float(np.mean(chunk)) if chunk else avg_energy
        else:
            local_e = avg_energy
        seg_energies.append(local_e)

    max_e = max(seg_energies) if seg_energies else (avg_energy or 1.0)

    for i, seg in enumerate(segments):
        local_e = seg_energies[i]
        rel_e = local_e / (max_e + 1e-6)
        st, et = seg["start_time"], seg["end_time"]

        if i == 0 and st < total_duration * 0.15 and rel_e < 0.7:
            section_type = "intro"
            section_label = "Intro / Atmosphere"
        elif i == num_segs - 1 and et > total_duration * 0.85 and rel_e < 0.75:
            section_type = "outro"
            section_label = "Outro / Fade"
        elif rel_e >= 0.85:
            section_type = "chorus_climax"
            section_label = "Chorus / Climax"
        elif i > 0 and seg_energies[i] > seg_energies[i - 1] * 1.15 and rel_e >= 0.65:
            section_type = "build_up"
            section_label = "Build-up / Escalation"
        elif i > 0 and seg_energies[i] < seg_energies[i - 1] * 0.75:
            section_type = "bridge"
            section_label = "Bridge / Breakdown"
        else:
            section_type = "verse"
            section_label = "Verse / Narrative"

        seg["section_type"] = section_type
        seg["section_label"] = section_label
        seg["local_energy"] = round(rel_e, 2)
        seg["emotional_tone"] = music.get("emotional_tone", "balanced_cinematic")


def _music_summary(music):
    if not music:
        return " Music    : not analysed"
    emo = music.get('emotional_tone', 'balanced_cinematic').replace('_', ' ')
    cnt = music.get('contrast_db', 0)
    return (f" Music    : {music['tempo']:.0f} BPM, {music['key']} {music['mode']}, "
            f"energy {music['energy']:.3f}, brightness {music['brightness_hz']:.0f} Hz, contrast {cnt:.1f} dB\n"
            f" Emotion  : {emo} (arousal {music.get('arousal', 0.5):.2f}, valence {music.get('valence', 0.0):+.2f})\n"
            f" Sounds   : {music['look'].replace('_', ' ')}  ->  suggests "
            f"'{music['suggested_preset']}' + {music['palette'][:44]}...")



# Cycled across cards so a lyric that spans several segments still produces
# visually different keyframes rather than a run of near-identical images.
SHOT_VARIATIONS = [
    "wide establishing shot",
    "medium shot",
    "close-up portrait",
    "low angle hero shot",
    "high angle overhead view",
    "over-the-shoulder shot",
    "side profile shot",
    "extreme close-up detail",
    "dutch angle shot",
    "sweeping aerial view",
]

MODEL_QUALITY_TOKENS = {
    "Z-Image": "sharp photographic detail, natural light falloff",
    "FLUX.1-Dev": "highly detailed, coherent composition, subtle film grain",
    "Qwen-Image": "crisp detail, balanced composition",
    "SDXL": "highly detailed, professional photography, 8k",
    "Custom": "",
}

# "Verse 1:", "[Chorus]", "(Bridge)" etc. are structure markers, not sung words.
SECTION_HEADER_RE = re.compile(
    r"^\s*[\[\(]?\s*"
    r"(verse|chorus|bridge|intro|outro|pre[\s-]?chorus|hook|refrain|interlude|breakdown|drop|solo|instrumental)"
    r"\b[\s\d.:_-]*[\]\)]?\s*:?\s*$",
    re.IGNORECASE,
)


def _clean_lyric_lines(lines):
    """Drop blank lines and section headers so cards are prompted with actual words."""
    cleaned = []
    for line in lines:
        text = str(line).strip()
        if not text or SECTION_HEADER_RE.match(text):
            continue
        cleaned.append(text)
    return cleaned


def _free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# If the decoder emits nothing for this long, the attempt is declared hung
# and abandoned. ctranslate2's CUDA build can deadlock inside cuDNN when it
# disagrees with the torch build loaded in the same process - on Windows this
# looks like "stuck on step 1 with no output at all".
WHISPER_STALL_TIMEOUT_SEC = 180.0


def _drain_whisper_segments(seg_iter, label=""):
    """
    Consume a faster-whisper segment generator on a worker thread so a native
    hang cannot freeze the whole queue: if no segment arrives for
    WHISPER_STALL_TIMEOUT_SEC the attempt raises TimeoutError and the caller
    moves on to the next compute type / backend. Also prints progress so the
    terminal shows the transcription is alive.
    """
    import threading
    import queue as _queue

    q = _queue.Queue()

    def worker():
        try:
            for seg in seg_iter:
                q.put(("seg", seg))
            q.put(("done", None))
        except BaseException as exc:  # noqa: BLE001 - forwarded to the main thread
            q.put(("err", exc))

    thread = threading.Thread(target=worker, daemon=True,
                              name="videoclipmaker-whisper-drain")
    thread.start()

    entries, count = [], 0
    while True:
        try:
            kind, payload = q.get(timeout=WHISPER_STALL_TIMEOUT_SEC)
        except _queue.Empty:
            raise TimeoutError(
                f"whisper {label} produced no output for {WHISPER_STALL_TIMEOUT_SEC:.0f}s "
                f"- the decoder appears hung (often a ctranslate2/cuDNN conflict); "
                f"abandoning this attempt")
        if kind == "done":
            break
        if kind == "err":
            raise payload
        seg = payload
        count += 1
        if count == 1 or count % 10 == 0:
            _safe_print(f"[AudioLyricsSegmenter]   ...transcribed up to "
                        f"{float(seg.end):.0f}s ({count} segments)")
        words = getattr(seg, "words", None)
        if words:
            for word in words:
                text = str(word.word).strip()
                if text:
                    entries.append({"start": float(word.start), "end": float(word.end), "text": text})
        else:
            text = str(seg.text).strip()
            if text:
                entries.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    return entries


def _register_cuda12_dlls():
    """
    ctranslate2 (faster-whisper's engine) is built against CUDA 12 and needs
    cublas64_12.dll / cudnn DLLs, while ComfyUI's torch cu130 ships only the
    CUDA 13 ones - without this the CUDA path fails with 'Library
    cublas64_12.dll is not found' or, worse, hangs silently inside cuDNN.
    The pip wheels nvidia-cublas-cu12 / nvidia-cudnn-cu12 carry the DLLs;
    Windows just never looks inside them unless we register the folders.
    """
    if os.name != "nt":
        return
    import importlib.util
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        try:
            spec = importlib.util.find_spec(pkg)
            for root in (spec.submodule_search_locations or []):
                bin_dir = os.path.join(root, "bin")
                if os.path.isdir(bin_dir):
                    # ctranslate2 resolves its CUDA DLLs through the PATH
                    # search, not add_dll_directory, so both are set.
                    os.add_dll_directory(bin_dir)
                    if bin_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass


def _transcribe_faster_whisper(audio_path, model_size, device, lyrics_hint=""):
    from faster_whisper import WhisperModel  # noqa: F401  (ImportError handled by caller)

    _register_cuda12_dlls()

    attempts = ([("cuda", "float16"), ("cuda", "int8_float16"), ("cuda", "int8"), ("cpu", "int8")]
                if device == "cuda" else [("cpu", "int8"), ("cpu", "float32")])

    last = None
    cuda_hung = False
    for dev, compute_type in attempts:
        if cuda_hung and dev == "cuda":
            continue  # a hung decoder hangs the same way at every precision
        model = None
        try:
            _safe_print(f"[AudioLyricsSegmenter] faster-whisper '{model_size}' on {dev} ({compute_type})...")
            model = WhisperModel(model_size, device=dev, compute_type=compute_type)
            # condition_on_previous_text=False stops the classic sung-repetition
            # spiral ("I want a roof" x80): once the decoder starts echoing its
            # own context on a looping chorus it never recovers. The thresholds
            # make it drop low-confidence hallucinated stretches instead of
            # keeping them. The typed lyrics, when given, seed the vocabulary so
            # "rule" is not misheard as "roof".
            seg_iter, _info = model.transcribe(
                audio_path, word_timestamps=True, vad_filter=True,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.0,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.5,
                initial_prompt=(lyrics_hint or None))
            entries = _drain_whisper_segments(seg_iter, label=f"({dev}, {compute_type})")
            return entries, f"faster-whisper/{model_size} ({dev}, {compute_type})"
        except Exception as exc:
            _safe_print(f"[AudioLyricsSegmenter] faster-whisper attempt ({dev}, {compute_type}) "
                        f"FAILED: {type(exc).__name__}: {exc}")
            if isinstance(exc, TimeoutError) and dev == "cuda":
                cuda_hung = True
                _safe_print("[AudioLyricsSegmenter] CUDA decoder hung - skipping remaining "
                            "CUDA precisions, falling back to CPU.")
            last = exc
        finally:
            del model
            _free_vram()
    raise last


def _transcribe_openai_whisper(audio_path, model_size, device, lyrics_hint=""):
    import whisper as openai_whisper  # noqa: F401  (ImportError handled by caller)

    model = None
    try:
        _safe_print(f"[AudioLyricsSegmenter] openai-whisper '{model_size}' on {device}...")
        model = openai_whisper.load_model(model_size, device=device)
        result = model.transcribe(audio_path, word_timestamps=True,
                                  condition_on_previous_text=False,
                                  initial_prompt=(lyrics_hint or None))

        entries = []
        for seg in result.get("segments", []):
            for word in (seg.get("words") or []):
                text = str(word.get("word", "")).strip()
                if text:
                    entries.append({"start": float(word["start"]), "end": float(word["end"]), "text": text})
            if not seg.get("words"):
                text = str(seg.get("text", "")).strip()
                if text:
                    entries.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": text})
        return entries, f"openai-whisper/{model_size} ({device})"
    finally:
        del model
        _free_vram()


def _transcribe_transformers(audio_path, model_size, device, lyrics_hint=""):
    """
    Works with the libraries ComfyUI already ships (transformers + torch + librosa),
    so lyric extraction needs no extra install.
    """
    from transformers import pipeline
    import librosa

    repo = f"openai/whisper-{model_size}"
    samples, _sr = librosa.load(audio_path, sr=16000, mono=True)

    dev_index = 0 if device == "cuda" else -1
    dtype = torch.float16 if device == "cuda" else torch.float32
    asr = None
    try:
        _safe_print(f"[AudioLyricsSegmenter] transformers '{repo}' on {device}...")
        try:
            asr = pipeline("automatic-speech-recognition", model=repo, device=dev_index, dtype=dtype)
        except TypeError:
            # transformers < 5 spells it torch_dtype
            asr = pipeline("automatic-speech-recognition", model=repo, device=dev_index, torch_dtype=dtype)

        result = asr({"raw": samples, "sampling_rate": 16000},
                     return_timestamps="word", chunk_length_s=30, batch_size=8)

        entries = []
        for chunk in result.get("chunks", []):
            text = str(chunk.get("text", "")).strip()
            stamp = chunk.get("timestamp") or (None, None)
            if not text or stamp[0] is None:
                continue
            start = float(stamp[0])
            end = float(stamp[1]) if stamp[1] is not None else start + 0.3
            entries.append({"start": start, "end": end, "text": text})
        return entries, f"transformers/{repo} ({device})"
    finally:
        del asr
        _free_vram()


def _separate_vocals(audio_path):
    """
    Isolate the vocal stem with demucs before transcription. Whisper mishears
    far less without drums and bass under the voice. Optional: if demucs is not
    installed or fails, the full mix is transcribed as before. Returns the path
    to a vocals-only WAV, or None.
    """
    try:
        from demucs.api import Separator
        import soundfile as sf
    except ImportError:
        return None

    separator = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _safe_print(f"[AudioLyricsSegmenter] demucs (htdemucs) isolating vocals on {device}...")
        separator = Separator(model="htdemucs", device=device)
        _origin, stems = separator.separate_audio_file(audio_path)
        vocals = stems.get("vocals")
        if vocals is None:
            return None
        out_dir = folder_paths.get_temp_directory() if folder_paths else os.path.abspath(".")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "videoclipmaker_vocals.wav")
        sf.write(out_path, vocals.transpose(0, 1).cpu().numpy(), separator.samplerate)
        _safe_print(f"[AudioLyricsSegmenter] Vocal stem -> {out_path}")
        return out_path
    except Exception as exc:
        _safe_print(f"[AudioLyricsSegmenter] demucs failed ({type(exc).__name__}: {exc}); "
                    "transcribing the full mix.")
        return None
    finally:
        del separator
        _free_vram()


def _collapse_repetitions(entries, max_repeats=4, max_phrase_len=8):
    """
    Whisper hallucinates on looping choruses: one sung "I wanna rule x4" comes
    back as the same phrase repeated eighty times, wall to wall. Real songs do
    repeat a hook, so the first `max_repeats` copies of any consecutive phrase
    (1..max_phrase_len words) are kept and the hallucinated tail is dropped.
    Timestamps of the surviving words are untouched.
    """
    words = [str(e["text"]).strip().lower().strip(".,!?\"'-") for e in entries]
    drop = [False] * len(entries)
    i = 0
    while i < len(entries):
        for n in range(1, max_phrase_len + 1):
            phrase = words[i:i + n]
            if len(phrase) < n:
                continue
            repeats = 1
            j = i + n
            while words[j:j + n] == phrase:
                repeats += 1
                j += n
            if repeats > max_repeats:
                for k in range(i + n * max_repeats, j):
                    drop[k] = True
                i = j - 1
                break
        i += 1
    kept = [e for e, d in zip(entries, drop) if not d]
    if len(kept) != len(entries):
        _safe_print(f"[AudioLyricsSegmenter] Collapsed {len(entries) - len(kept)} hallucinated "
                    f"repeated word(s) out of {len(entries)}.")
    return kept


def _transcribe_song(audio_path, model_size, lyrics_hint=""):
    """
    Real lyric extraction with word-level timestamps.
    Returns (entries, backend_label). On total failure returns ([], None) with a loud
    warning rather than raising - a missing optional backend must not kill the workflow.
    `lyrics_hint` (the typed custom_lyrics, when present) seeds the decoder
    vocabulary so unusual words are heard as written.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backends = (
        ("faster-whisper", _transcribe_faster_whisper),
        ("openai-whisper", _transcribe_openai_whisper),
        ("transformers", _transcribe_transformers),
    )

    hint = " ".join(str(lyrics_hint or "").split())[:800]

    # Transcribe the isolated vocal stem when demucs is available - the
    # timestamps stay valid because the stem is sample-aligned with the mix.
    vocals_path = _separate_vocals(audio_path)
    transcribe_path = vocals_path or audio_path

    problems = []
    for name, fn in backends:
        try:
            entries, label = fn(transcribe_path, model_size, device, lyrics_hint=hint)
            if vocals_path:
                label += " +demucs vocals"
            return _collapse_repetitions(entries), label
        except ImportError:
            problems.append(f"{name}: not installed")
        except Exception as exc:
            problems.append(f"{name}: {type(exc).__name__}: {exc}")

    _safe_print(
        "[AudioLyricsSegmenter] " + "!" * 60 + "\n"
        f"[AudioLyricsSegmenter] Could not transcribe lyrics with whisper_mode='{model_size}':\n  "
        + "\n  ".join(problems)
        + "\n[AudioLyricsSegmenter] Falling back to the text in 'custom_lyrics'.\n"
          "[AudioLyricsSegmenter] For real transcription install a backend into ComfyUI's python, e.g.\n"
          "[AudioLyricsSegmenter]   python_embeded\\python.exe -m pip install faster-whisper\n"
        + "[AudioLyricsSegmenter] " + "!" * 60
    )
    return [], None


def _assign_transcript_to_segments(entries, segments):
    """
    Put each transcribed word into the video segment its midpoint falls in, so a card's
    prompt carries the words actually sung during that clip. Mutates `segments`.
    """
    for seg in segments:
        start_t, end_t = seg["start_time"], seg["end_time"]
        picked = [e["text"] for e in entries
                  if start_t <= (e["start"] + e["end"]) / 2.0 < end_t]
        if not picked:
            # Segment-level entries can straddle a window; fall back to any overlap.
            picked = [e["text"] for e in entries if e["end"] > start_t and e["start"] < end_t]
        text = " ".join(picked).split()
        seg["lyrics"] = " ".join(text) if text else INSTRUMENTAL
    return segments


def _distribute_lyrics(lines, num_segments):
    """
    Spread lyric lines evenly across every segment.
    Fewer lines than segments -> each line covers a contiguous run of segments
    (never dumps the final line onto the whole tail).
    More lines than segments -> segments carry several lines each.
    """
    lines = _clean_lyric_lines(lines)
    if not lines:
        return [INSTRUMENTAL] * num_segments
    if num_segments <= 0:
        return []

    out = []
    if len(lines) >= num_segments:
        per = len(lines) / float(num_segments)
        for i in range(num_segments):
            start_i = int(round(i * per))
            end_i = max(int(round((i + 1) * per)), start_i + 1)
            out.append(" / ".join(lines[start_i:end_i]))
    else:
        for i in range(num_segments):
            idx = min(len(lines) - 1, int(i * len(lines) / num_segments))
            out.append(lines[idx])
    return out


def _fmt_time(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:05.2f}"


def _parse_durations(text):
    """Accepts '6, 6, 8' / newline separated / space separated. Ignores junk tokens."""
    if not text or not text.strip():
        return []
    values = []
    for token in re.split(r"[,;\s]+", text.strip()):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _build_segments(total_duration, durations, lines, fallback_duration):
    """
    Lay segments end to end across the song. `durations` are user overrides;
    anything beyond the list falls back to `fallback_duration`.
    Returns (segments, notes).
    """
    notes = []
    spans = []
    cursor = 0.0
    index = 0

    while cursor < total_duration - 0.01:
        if index < len(durations):
            requested = durations[index]
            clamped = max(MIN_SEGMENT_SEC, min(MAX_SEGMENT_SEC, requested))
            if abs(clamped - requested) > 0.001:
                notes.append(
                    f"segment {index}: {requested:.2f}s is outside the "
                    f"{MIN_SEGMENT_SEC:g}-{MAX_SEGMENT_SEC:g}s rule -> clamped to {clamped:.2f}s"
                )
        else:
            clamped = fallback_duration

        end = min(total_duration, cursor + clamped)
        spans.append((cursor, end))
        cursor = end
        index += 1

    if not spans:
        spans = [(0.0, total_duration)]

    # A leftover tail shorter than the minimum gets folded into the previous segment.
    if len(spans) > 1 and (spans[-1][1] - spans[-1][0]) < MIN_SEGMENT_SEC:
        tail_start, tail_end = spans.pop()
        prev_start, _ = spans[-1]
        spans[-1] = (prev_start, tail_end)
        notes.append(
            f"final tail of {tail_end - tail_start:.2f}s was shorter than "
            f"{MIN_SEGMENT_SEC:g}s -> merged into the previous segment"
        )

    if len(durations) > len(spans):
        notes.append(
            f"{len(durations)} durations given but only {len(spans)} fit in the song -> extras ignored"
        )

    lyric_texts = _distribute_lyrics(lines, len(spans))
    segments = []
    for i, (start_t, end_t) in enumerate(spans):
        segments.append({
            "segment_index": i,
            "start_time": round(start_t, 2),
            "end_time": round(end_t, 2),
            "duration": round(end_t - start_t, 2),
            "lyrics": lyric_texts[i],
        })
    return segments, notes


def _wrap(text, width, indent):
    """Wrap long lyrics onto continuation lines instead of cutting them off -
    the whole point of the table is reading what is actually sung."""
    words, lines, current = str(text).split(), [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if not lines:
        return ""
    pad = "\n" + " " * indent
    return pad.join(lines)


def _lyrics_sheet(data):
    """The complete transcript laid out against the timeline, nothing truncated."""
    segments = data.get("segments", [])
    total = float(data.get("total_duration", 0.0))
    transcript = data.get("transcript") or []
    words = sum(len(str(s.get("lyrics", "")).split())
                for s in segments if s.get("lyrics") != INSTRUMENTAL)

    bar = "=" * 78
    rows = [bar,
            " FULL LYRICS - mapped onto the song timeline",
            bar,
            f" Source   : {data.get('lyric_source', 'custom_lyrics')}",
            f" Duration : {_fmt_time(total)}   Segments: {len(segments)}   Cards: {len(segments) + 1}",
            f" Words    : {words} across {sum(1 for s in segments if s.get('lyrics') != INSTRUMENTAL)} "
            f"singing segment(s)"
            + (f", from {len(transcript)} timed tokens" if transcript else ""),
            bar, ""]

    for seg in segments:
        rows.append(f"[{seg['segment_index']:03d}]  {_fmt_time(seg['start_time'])} - "
                    f"{_fmt_time(seg['end_time'])}   ({seg['duration']:.2f}s)")
        text = seg.get("lyrics", "")
        if text == INSTRUMENTAL:
            rows.append("       (instrumental - no vocals in this segment)")
        else:
            for line in _wrap(text, 68, 0).split("\n"):
                rows.append("       " + line)
        rows.append("")

    rows.append(bar)
    return "\n".join(rows)


def _timeline_table(data, title, approved, notes, hint_lines):
    segments = data.get("segments", [])
    total = data.get("total_duration", 0.0)
    status = "APPROVED - running next step" if approved else "NOT APPROVED - workflow paused here"
    bar = "=" * 78

    rows = [
        bar,
        f" {title}   [ {status} ]",
        bar,
        f" Song file : {os.path.basename(str(data.get('audio_path', 'n/a')))}",
        f" Duration  : {_fmt_time(total)}  ({total:.2f} s)",
        f" Segments  : {len(segments)}   ->  storyboard cards needed: {len(segments) + 1}",
        f" Rule      : every segment must be {MIN_SEGMENT_SEC:g}-{MAX_SEGMENT_SEC:g} s",
        _music_summary(data.get("music")),
        f" Style     : {data.get('style_preset', '-')}",
        f" Applied   : {_wrap(data.get('global_prompt') or '(none)', 58, 13)}",
        f" Lyrics    : {data.get('lyric_source', 'custom_lyrics')}"
        f"   ({sum(1 for s in segments if s.get('lyrics') != INSTRUMENTAL)}/{len(segments)} segments with words)",
        "-" * 78,
        " IDX   START      END        DUR      LYRICS",
    ]
    for seg in segments:
        flag = " " if MIN_SEGMENT_SEC - 0.01 <= seg["duration"] <= MAX_SEGMENT_SEC + 0.01 else "!"
        rows.append(
            f" {seg['segment_index']:03d}{flag} {_fmt_time(seg['start_time'])}   "
            f"{_fmt_time(seg['end_time'])}   {seg['duration']:5.2f}s   {_wrap(seg['lyrics'], 44, 46)}"
        )

    if notes:
        rows.append("-" * 78)
        rows.append(" NOTES:")
        rows.extend(f"  - {n}" for n in notes)

    rows.append("-" * 78)
    rows.extend(hint_lines)
    rows.append(bar)
    return "\n".join(rows)


class DynamicCardBatchPrompter:
    """
    Dynamically extracts prompts for all N+1 storyboard cards generated for any song duration.
    Allows image generators (Z-Image / FLUX / SDXL) to generate all storyboard card images dynamically.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_rerun": FORCE_RERUN_INPUT,
                "prompt_list": ("STRING", {"forceInput": True}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1,
                                        "control_after_generate": True}),
            },
            "optional": {
                "rerender_card": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1,
                                          "tooltip": "-1 = render the sequence. Set a card number to redo just that one."}),
                "override_prompt": ("STRING", {"multiline": True, "default": "",
                                               "placeholder": "Optional: replace the prompt for the card being rendered."}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("CURRENT_PROMPT", "CARD_INDEX", "TOTAL_CARDS", "HAS_NEXT")
    FUNCTION = "get_prompt"
    CATEGORY = "Geekatplay/VideoClipMaker"

    def get_prompt(self, prompt_list, batch_index=0, rerender_card=-1, override_prompt="",
                   force_rerun=False):
        prompts = json.loads(prompt_list)
        total = len(prompts)
        if total == 0:
            return ("", 0, 0, False)

        # A card number pins the render to that one card; -1 walks the sequence.
        wanted = rerender_card if rerender_card is not None and rerender_card >= 0 else batch_index
        idx = max(0, min(int(wanted), total - 1))

        current_prompt = override_prompt.strip() if override_prompt.strip() else prompts[idx]
        has_next = (idx < total - 1)

        mode = "rerender" if rerender_card is not None and rerender_card >= 0 else "sequence"
        note = " (prompt overridden)" if override_prompt.strip() else ""
        _safe_print(f"[DynamicCardBatchPrompter] Card {idx + 1}/{total} [{mode}]{note}")
        return (current_prompt, idx, total, has_next)


# --------------------------------------------------------------------------
# Runs. Every approved Part 1 writes a fresh timestamped folder holding that
# run's project.json and its rendered cards, so nothing overwrites an earlier
# attempt and Parts 2/3 can be pointed at whichever run you want.

RUN_LATEST = "<< newest run >>"


def _runs_root():
    base = folder_paths.get_output_directory() if folder_paths else os.path.abspath("./output")
    root = os.path.join(base, "storyboard_projects")
    os.makedirs(root, exist_ok=True)
    return root


def _safe_name(text, fallback="music_video"):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip()).strip("_") or fallback


def _list_runs():
    """Existing run folder names, newest first."""
    root = _runs_root()
    try:
        entries = [e for e in os.listdir(root)
                   if os.path.isfile(os.path.join(root, e, "project.json"))]
    except OSError:
        return []
    entries.sort(key=lambda e: os.path.getmtime(os.path.join(root, e, "project.json")),
                 reverse=True)
    return entries


def _resolve_run(run):
    """Turn a dropdown choice into a real run folder."""
    runs = _list_runs()
    if run in (None, "", RUN_LATEST):
        if not runs:
            raise ValueError(
                "No saved runs yet.\nRun Part 1, set 'approve_timeline' to APPROVED and queue - "
                "that writes output/storyboard_projects/<project>__<timestamp>/project.json."
            )
        return os.path.join(_runs_root(), runs[0])

    candidate = os.path.join(_runs_root(), run)
    if os.path.isfile(os.path.join(candidate, "project.json")):
        return candidate
    raise ValueError(
        f"Run '{run}' not found.\nAvailable runs: {runs or '(none)'}\n"
        "Pick one from the dropdown, or re-approve Part 1 to create a new run. "
        "If a brand-new run is missing from the list, press 'refresh run list' on this node."
    )


def _new_run_dir(project_name):
    """A fresh, uniquely identified folder for every run.

    The timestamp alone is not an identity - two queues inside the same second,
    or two projects sharing a name, would land in the same folder and overwrite
    each other's cards, clips and song. The short id makes every run its own
    thing, so nothing a previous run produced can ever be written over.
    """
    import datetime, uuid
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    while True:
        run_id = uuid.uuid4().hex[:6]
        path = os.path.join(_runs_root(), f"{_safe_name(project_name)}__{stamp}__{run_id}")
        if not os.path.exists(path):
            break
    os.makedirs(os.path.join(path, "cards"), exist_ok=True)
    return path, run_id


def _run_key(project_name, storyboard_data):
    """A stable id for 'this song, this timeline, this style'."""
    import hashlib
    return hashlib.sha1(
        (str(project_name) + "|" + str(storyboard_data)).encode("utf-8")).hexdigest()[:6]


def _stable_run_dir(project_name, run_id):
    """The run folder for a given key, reused if it already exists.

    The all-in-one workflow queues three times for ONE project (approve the
    timeline, approve the cards, render). A new folder per queue would strand
    the cards in the previous one, so there the run is identified by its
    content: same song + timeline + style -> same folder, every time.
    """
    root = _runs_root()
    safe = _safe_name(project_name)
    suffix = "__" + run_id
    try:
        for entry in os.listdir(root):
            if entry.startswith(safe + "__") and entry.endswith(suffix):
                path = os.path.join(root, entry)
                os.makedirs(os.path.join(path, "cards"), exist_ok=True)
                return path, run_id
    except OSError:
        pass
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, f"{safe}__{stamp}{suffix}")
    os.makedirs(os.path.join(path, "cards"), exist_ok=True)
    return path, run_id


def _run_cards_dir(run_dir):
    path = os.path.join(run_dir, "cards")
    os.makedirs(path, exist_ok=True)
    return path


def _view_ref(path):
    """
    Describe a file inside the run so the canvas can display it.

    ComfyUI serves anything under its output folder from /api/view, so a card
    or a finished clip only needs its name, its subfolder and the folder type -
    that is what lets the progress board show the real first/last frames and
    play back each segment as it lands, instead of describing them in text.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        root = (folder_paths.get_output_directory() if folder_paths
                else os.path.abspath("./output"))
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
        if rel.startswith(".."):
            return None                      # outside output/: not servable
        rel = rel.replace("\\", "/")
        return {"filename": os.path.basename(rel),
                "subfolder": os.path.dirname(rel),
                "type": "output"}
    except Exception:
        return None


def _run_references_dir(run_dir):
    path = os.path.join(run_dir, "references")
    os.makedirs(path, exist_ok=True)
    return path


def _project_file(project_name):
    base = folder_paths.get_output_directory() if folder_paths else os.path.abspath("./output")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(project_name).strip()) or "music_video"
    folder = os.path.join(base, "storyboard_projects")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{safe}.json")


# Ready-made looks for the whole video. Picked in Part 1 and merged into every
# card prompt; "Custom" leaves it entirely to your own description.
STYLE_PRESETS = {
    "Auto (match the music)": "",
    "Custom (use my description only)": "",
    "Photorealistic": "photorealistic, natural lighting, true-to-life colour, "
                      "fine skin and fabric detail, realistic optics",
    "Cinematic film": "cinematic film still, shot on 35mm, anamorphic lens, "
                      "professional colour grading, shallow depth of field",
    "Pencil drawing": "hand-drawn graphite pencil sketch, visible hatching and smudged shading, "
                      "paper grain, monochrome",
    "Ink and watercolour": "ink linework with loose watercolour washes, bleeding pigment, "
                           "textured paper",
    "Oil painting": "oil painting, thick impasto brushstrokes, canvas texture, "
                    "rich pigment, painterly edges",
    "Rococo": "rococo painting, ornate gilded detail, pastel palette, soft diffused light, "
              "elaborate 18th century decoration",
    "Art nouveau": "art nouveau illustration, flowing organic linework, decorative borders, "
                   "flat stylised colour",
    "Abstract": "abstract non-representational composition, bold shapes and colour fields, "
                "gestural marks, minimal literal detail",
    "Anime / cel": "anime cel animation, clean bold linework, flat cel shading, "
                   "expressive stylised features",
    "Comic book": "comic book art, heavy inked outlines, halftone dot shading, "
                  "dynamic action panel composition",
    "3D render": "high-end 3D render, physically based materials, ray-traced global illumination, "
                 "subtle ambient occlusion",
    "Claymation": "stop-motion clay animation, visible fingerprints in the clay, "
                  "handmade miniature set, tactile texture",
    "Film noir": "black and white film noir, hard chiaroscuro lighting, venetian blind shadows, "
                 "heavy grain, high contrast",
    "Cyberpunk": "cyberpunk aesthetic, dense neon signage, wet reflective streets, "
                 "holographic advertising, moody atmosphere",
    "Vintage retro": "1970s film photography, faded warm colour, halation, "
                     "soft focus, light leaks",
    "Action blockbuster": "high-energy action cinematography, dynamic motion blur, "
                          "practical explosions and sparks, dramatic low angles",
    "Surreal dreamlike": "surreal dreamlike imagery, impossible scale and physics, "
                         "soft glowing haze, symbolic composition",
}


def _compose_style(preset, description, music=None):
    """
    Preset tokens first, then whatever you typed - both reach every card.
    'Auto' takes its look straight from how the song sounds.
    """
    if preset == "Auto (match the music)" and music:
        base = STYLE_PRESETS.get(music.get("suggested_preset", ""), "")
        parts = [base.strip(), (music.get("palette") or "").strip(),
                 (description or "").strip()]
    else:
        parts = [STYLE_PRESETS.get(preset, "").strip(), (description or "").strip()]
    return ", ".join(p.rstrip(",. ") for p in parts if p)


class StoryboardProjectSave:
    """
    Writes the approved timeline to disk so the next workflow can pick it up.
    This is what lets part 1 / part 2 / part 3 be run as separate graphs.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard_data": ("STRING", {"forceInput": True}),
                "project_name": ("STRING", {"default": "music_video"}),
                "run_mode": (["new run every queue",
                              "one run per song + timeline"], {
                    "default": "new run every queue",
                    "tooltip": "New run every queue: each queue of Part 1 writes its own "
                               "project folder. One run per song + timeline: the same song, "
                               "timeline and style always resolve to the SAME folder - "
                               "required by the all-in-one workflow, which queues three times "
                               "for one project.",
                }),
            },
            "optional": {
                "prompt_list": ("STRING", {"forceInput": True}),
                "animation_prompts": ("STRING", {"forceInput": True}),
                "reference_plan": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("PROJECT_FILE", "RUN_DIR", "AUDIO_PATH", "STORYBOARD_DATA")
    FUNCTION = "save_project"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = "Saves the approved timeline so the next workflow can load it."

    @classmethod
    def IS_CHANGED(cls, run_mode="new run every queue", project_name="music_video",
                   storyboard_data="", **kwargs):
        # Never cache in "new run every queue" mode: re-queueing with unchanged
        # settings would otherwise hit ComfyUI's cache, the node would silently
        # not execute, and no new project would appear - which looks exactly
        # like the save being broken.
        if str(run_mode).startswith("new run"):
            return float("nan")
        return _run_key(project_name, storyboard_data)

    def save_project(self, storyboard_data, project_name="music_video",
                     prompt_list=None, animation_prompts=None, reference_plan=None,
                     run_mode="new run every queue"):
        data = json.loads(storyboard_data)
        # Part 1 always leaves you a project to open, and every queue gets its own
        # uniquely identified folder - approved or not. Refusing to save until a
        # toggle is flipped just looks like the step failed, and reusing a folder
        # would let one run overwrite another run's cards, clips and song.
        approved = bool(data.get("approved", True))
        if str(run_mode).startswith("new run"):
            run_dir, run_id = _new_run_dir(project_name)
        else:
            run_dir, run_id = _stable_run_dir(project_name,
                                              _run_key(project_name, storyboard_data))
        data["project_name"] = project_name
        data["run_dir"] = run_dir
        data["run_id"] = run_id
        data["approved"] = approved

        # Keep the ORIGINAL song with the run. Audio handed over by a Load Audio
        # node lives in ComfyUI/temp, which is wiped between sessions - without
        # this copy the final video would later be muxed against a missing file.
        source_audio = data.get("audio_path") or ""
        if source_audio and os.path.isfile(source_audio):
            kept = os.path.join(run_dir, "song" + os.path.splitext(source_audio)[1].lower())
            try:
                if os.path.abspath(source_audio) != os.path.abspath(kept):
                    shutil.copy2(source_audio, kept)
                data["audio_path"] = kept
                data["original_audio"] = source_audio
            except Exception as exc:
                _safe_print(f"[StoryboardProjectSave] Could not copy the song into the run "
                            f"({exc}); keeping the original path.")
        elif source_audio:
            _safe_print(f"[StoryboardProjectSave] WARNING: song not found at {source_audio} - "
                        "the final video will have no audio unless you re-run Part 1.")
        # The finished prompts ride along, so Parts 2 and 3 render exactly what you
        # reviewed here instead of rebuilding them.
        if prompt_list:
            data["prompt_list"] = json.loads(prompt_list)
        if animation_prompts:
            data["animation_prompts"] = json.loads(animation_prompts)
        if reference_plan:
            data["reference_plan"] = json.loads(reference_plan)
        path = os.path.join(run_dir, "project.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

        segments = data.get("segments", [])
        report = (f"SAVED RUN  {os.path.basename(run_dir)}\n"
                  f"run id : {run_id}   ({'APPROVED' if approved else 'not approved yet'})\n"
                  f"{path}\n"
                  f"{len(segments)} segments, {data.get('card_count', len(segments) + 1)} cards, "
                  f"lyrics from {data.get('lyric_source', 'custom_lyrics')}.\n"
                  f"Style: {data.get('style_preset', '-')}\n"
                  f"{len(data.get('prompt_list') or [])} card prompts stored.\n"
                  f"song kept at: {data.get('audio_path')}\n\n"
                  f"In Part 2, leave the 'run' dropdown on '{RUN_LATEST}' - it already "
                  f"points here. To pick this run by name, press 'refresh run list' on "
                  f"the loader node first.")
        if not approved:
            report += ("\n\nEvery queue writes its own run, so nothing here will be "
                       "overwritten by the next one.\nFlip 'approve_timeline' to APPROVED "
                       "once the timeline is right - that marks the run approved and keeps "
                       "the panels honest about which one you settled on.")
        _safe_print(f"[StoryboardProjectSave] {report}")
        # AUDIO_PATH is the copy kept inside the run, so the final mux never
        # depends on ComfyUI's temp folder. STORYBOARD_DATA carries the run id
        # and paths, so the all-in-one workflow needs no project loader.
        return {"ui": {"text": (report,)},
                "result": (path, run_dir, data.get("audio_path", ""),
                           json.dumps(data, indent=2))}


class StoryboardProjectLoad:
    """Loads one saved run, so each stage can be run on its own."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run": ([RUN_LATEST] + _list_runs(), {"default": RUN_LATEST}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("STORYBOARD_DATA", "PROMPT_LIST", "ANIMATION_PROMPTS", "REFERENCE_PLAN",
                    "RUN_DIR", "AUDIO_PATH", "FULL_LYRICS", "SEGMENT_COUNT")
    FUNCTION = "load_project"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = "Loads the timeline saved by workflow 1."

    @classmethod
    def IS_CHANGED(cls, run=RUN_LATEST, **kwargs):
        # Always re-read: "newest run" must follow a fresh Part 1 approval.
        try:
            path = os.path.join(_resolve_run(run), "project.json")
            return f"{path}:{os.path.getmtime(path)}"
        except Exception:
            return float("nan")

    def load_project(self, run=RUN_LATEST):
        run_dir = _resolve_run(run)
        path = os.path.join(run_dir, "project.json")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        # Older runs were written before some fields existed - fill them in so the
        # rest of the pipeline can rely on them.
        data.setdefault("style_preset", "-")
        data.setdefault("global_prompt", "")
        data.setdefault("card_count", len(data.get("segments", [])) + 1)
        data["run_dir"] = run_dir

        segments = data.get("segments", [])
        # Older runs stored a temp path; prefer the copy that lives in the run.
        for candidate in (os.path.join(run_dir, "song.wav"), os.path.join(run_dir, "song.mp3"),
                          os.path.join(run_dir, "song.flac")):
            if os.path.isfile(candidate):
                data["audio_path"] = candidate
                break

        prompts = data.get("prompt_list") or []
        motions = data.get("animation_prompts") or []
        if not prompts:
            raise ValueError(
                f"Run '{os.path.basename(run_dir)}' has no card prompts saved - it was written "
                "by an older build.\nRe-run Part 1 and approve again; prompts are stored with "
                "the timeline now."
            )

        state = "APPROVED" if data.get("approved", True) else "not approved in Part 1"
        report = (f"LOADED RUN  {os.path.basename(run_dir)}\n"
                  f"run id : {data.get('run_id', '-')}   ({state})\n"
                  f"{_fmt_time(float(data.get('total_duration', 0.0)))} song, {len(segments)} segments, "
                  f"{data.get('card_count', len(segments) + 1)} cards.\n"
                  f"Style: {data.get('style_preset', '-')}\n"
                  f"{len(prompts)} card prompts, {len(motions)} motion prompts.\n"
                  f"cards -> {os.path.join(run_dir, 'cards')}\n"
                  + (f"song  -> {data.get('audio_path')}"
                     if os.path.isfile(data.get("audio_path") or "")
                     else "song  -> MISSING (final video would have no audio; re-run Part 1)"))
        _safe_print(f"[StoryboardProjectLoad] {report}")
        return {"ui": {"text": (report + "\n\n" + _lyrics_sheet(data),)},
                "result": (json.dumps(data, indent=2),
                           json.dumps(prompts, indent=2), json.dumps(motions, indent=2),
                           json.dumps(data.get("reference_plan") or
                                      {"references": [], "assignment": {}}, indent=2), run_dir,
                           data.get("audio_path", ""),
                           _lyrics_sheet(data), len(segments))}


class MusicVideoStatus:
    """
    Always-on dashboard: which step the pipeline is on, what is finished, what to do
    next, and which node's force_rerun to flip to redo a step. Never cached, so it
    refreshes on every queue even when every other step is cached.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_folder": ("STRING", {"default": "video"}),
                "filename_filter": ("STRING", {"default": "LTX-2.5_i2v"}),
                "output_filename": ("STRING", {"default": "final_music_video.mp4"}),
            },
            "optional": {
                "storyboard_data": ("STRING", {"forceInput": True}),
                "run_dir": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("STATUS", )
    FUNCTION = "report"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = "Shows which step the music video pipeline is on and what to do next."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")   # always refresh - it is only filesystem stats

    @staticmethod
    def _bar(done, total, width=34):
        if total <= 0:
            return "-" * width
        filled = int(round(width * min(done, total) / total))
        return "#" * filled + "." * (width - filled)

    def report(self, segment_folder="video",
               filename_filter="LTX-2.5_i2v", output_filename="final_music_video.mp4",
               storyboard_data=None, run_dir=None):
        out_dir = folder_paths.get_output_directory() if folder_paths else os.path.abspath("./output")

        data, segments, required = None, [], 0
        if storyboard_data:
            try:
                data = json.loads(storyboard_data)
                segments = data.get("segments", [])
                required = data.get("card_count", len(segments) + 1)
            except Exception:
                data = None

        # --- what exists on disk right now ---
        runs = _list_runs()
        if not run_dir and data:
            run_dir = data.get("run_dir")
        cards = 0
        if run_dir and os.path.isdir(run_dir):
            cards = len([i for i, _ in _card_files(_run_cards_dir(run_dir))
                         if required == 0 or i < required])

        clips_dir = segment_folder if os.path.isabs(segment_folder) else os.path.join(out_dir, segment_folder)
        clips = 0
        if os.path.isdir(clips_dir):
            name_filter = (filename_filter or "").strip()
            clips = len([f for f in os.listdir(clips_dir)
                         if f.lower().endswith((".mp4", ".mkv", ".webm"))
                         and (not name_filter or f.startswith(name_filter))
                         and f != output_filename])

        final_path = os.path.join(out_dir, output_filename)
        final_done = os.path.isfile(final_path)

        # --- which step are we on ---
        if not data:
            step = 1
        elif cards < required:
            step = 2
        elif clips < len(segments):
            step = 3
        elif not final_done:
            step = 4
        else:
            step = 5

        def mark(n):
            return "-->" if n == step else (" ok" if n < step else "   ")

        bar = "=" * 78
        rows = [bar, " MUSIC VIDEO - PIPELINE STATUS", bar]

        if data:
            total = float(data.get("total_duration", 0.0))
            with_words = sum(1 for s in segments if s.get("lyrics") != INSTRUMENTAL)
            rows += [
                f" Run     : {os.path.basename(run_dir) if run_dir else '(not saved yet)'}",
                f" Song    : {_fmt_time(total)} ({total:.2f}s) -> {len(segments)} segments, {required} cards",
                f" Lyrics  : {data.get('lyric_source', 'custom_lyrics')}  "
                f"({with_words}/{len(segments)} segments with words)",
            ]
        else:
            rows.append(" Run     : (waiting for the song to be analysed)")
        rows.append(f" Saved runs: {len(runs)}" + (f"   newest: {runs[0]}" if runs else ""))

        rows.append("-" * 78)
        rows.append(f" {mark(1)} STEP 1  Song timeline    "
                    + (f"{len(segments)} segments mapped" if data else "not analysed yet"))
        rows.append(f" {mark(2)} STEP 2  Storyboard deck  {cards}/{required} cards   "
                    f"[{self._bar(cards, required)}]" if required else
                    f" {mark(2)} STEP 2  Storyboard deck  waiting")
        rows.append(f" {mark(3)} STEP 3  Video segments   {clips}/{len(segments)} clips  "
                    f"[{self._bar(clips, len(segments))}]" if segments else
                    f" {mark(3)} STEP 3  Video segments   waiting")
        rows.append(f" {mark(4)} STEP 4  Final video      "
                    + (f"built -> {output_filename}" if final_done else "not built yet"))
        rows.append("-" * 78)

        if step == 1:
            rows += [" NEXT: pick a song in 'Load Audio' and queue.",
                     "       Then read the timeline table and flip 'approve_timeline' on."]
        elif step == 2:
            missing = required - cards
            rows += [f" NEXT: {missing} card(s) still to render.",
                     f"       Set the Queue batch count to {missing} and queue once -",
                     "       'card_index' auto-increments so each run renders the next card.",
                     "       Redo one card: set 'rerender_card' to its number, queue, set back to -1."]
        elif step == 3:
            missing = len(segments) - clips
            queues = -(-missing // 4)   # the render node's default segments_per_queue
            rows += [" All cards are rendered. Flip 'approve_cards' on if you have not yet.",
                     f" NEXT: {missing} video segment(s) still to render.",
                     f"       Each queue renders a batch (segments_per_queue, default 4), so",
                     f"       set the Queue batch count to ~{queues} and queue once -",
                     "       finished clips are skipped, extra queues just do nothing."]
        elif step == 4:
            rows += [" NEXT: every segment is rendered. Queue once more to stitch them",
                     "       together with the song into the final video."]
        else:
            rows += [f" DONE: {final_path}"]

        rows += ["-" * 78,
                 " RERUN A STEP: flip 'force_rerun' (top toggle) on that node, queue, flip it off.",
                 "   step 1 -> Audio Lyrics Segmenter    re-transcribe / re-split the song",
                 "   step 1 -> Song Timeline Review      re-apply segment retiming",
                 "   step 2 -> Storyboard Prompt Gen     rebuild every card prompt",
                 "   step 2 -> Card Batch Prompter       re-render the current card",
                 "   step 3 -> Keyframe Pair Batcher     re-render the current segment",
                 "   step 4 -> Video Segment Stitcher    rebuild the final video",
                 _gap_credit(), bar]

        status = "\n".join(rows)
        _safe_print(status)
        return {"ui": {"text": (status,)}, "result": (status,)}


def _cards_dir(project_name):
    base = folder_paths.get_output_directory() if folder_paths else os.path.abspath("./output")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(project_name).strip()) or "music_video"
    path = os.path.join(base, "storyboard_cards", safe)
    os.makedirs(path, exist_ok=True)
    return path


def _card_files(folder):
    """Return [(index, path)] for card_XXXX.png files, sorted by index."""
    found = []
    for name in os.listdir(folder):
        match = re.match(r"^card_(\d+)\.png$", name, re.IGNORECASE)
        if match:
            found.append((int(match.group(1)), os.path.join(folder, name)))
    return sorted(found)


class StoryboardCardSaver:
    """
    Writes each rendered storyboard card to the project's card folder as card_XXXX.png,
    building the deck up one queue at a time. No pre-existing files are needed.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "card_index": ("INT", {"default": 0, "min": 0, "max": 9999,
                                       "tooltip": "Index of the first image in this batch. 0 when saving a whole deck."}),
                "storyboard_data": ("STRING", {"forceInput": True}),
                "run_dir": ("STRING", {"forceInput": True}),
                "filename": ("STRING", {
                    "default": "",
                    "tooltip": "Leave empty to save numbered cards. Set e.g. 'reference' to "
                               "store a single named image instead.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("SAVE_REPORT", )
    FUNCTION = "save_card"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = "Saves the rendered keyframe card for this index into the project card folder."

    def save_card(self, images, card_index, storyboard_data, run_dir, filename=""):
        from PIL import Image

        data = json.loads(storyboard_data)
        required = data.get("card_count", len(data.get("segments", [])) + 1)
        folder = _run_cards_dir(run_dir)

        named = (filename or "").strip()
        if named:
            # a single named image (the character reference), kept beside the deck
            array = np.clip(images[0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            path = os.path.join(run_dir, _safe_name(named, "reference") + ".png")
            Image.fromarray(array).save(path, compress_level=4)
            report = f"Saved {os.path.basename(path)} -> {run_dir}"
            _safe_print(f"[StoryboardCardSaver] {report}")
            return {"ui": {"text": (report,)}, "result": (report,)}

        written = []
        for offset in range(images.shape[0]):
            index = int(card_index) + offset
            array = images[offset].detach().cpu().numpy()
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            path = os.path.join(folder, f"card_{index:04d}.png")
            Image.fromarray(array).save(path, compress_level=4)
            written.append(index)

        have = len(_card_files(folder))
        report = (f"Saved card(s) {written} -> {folder}\n"
                  f"Deck progress: {have}/{required} cards rendered.")
        if have < required:
            report += f"\nStill missing {required - have}. Queue again (card_index auto-increments)."
        else:
            report += "\nDeck complete - approve the cards to start video generation."
        _safe_print(f"[StoryboardCardSaver] {report}")
        return {"ui": {"text": (report,)}, "result": (report,)}


class StoryboardCardLoader:
    """
    Assembles every card rendered so far into one IMAGE batch, so the deck is built
    dynamically inside the workflow instead of loaded from files prepared beforehand.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_rerun": FORCE_RERUN_INPUT,
                "storyboard_data": ("STRING", {"forceInput": True}),
                "run_dir": ("STRING", {"forceInput": True}),
                "require_complete": ("BOOLEAN", {"default": True,
                                                 "label_on": "wait for every card",
                                                 "label_off": "load whatever exists"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("CARD_IMAGES", "CARDS_FOUND", "DECK_REPORT")
    FUNCTION = "load_cards"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = "Loads all storyboard cards rendered so far into a single batch."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The folder fills up as cards render, so hash its contents, not the widgets.
        if kwargs.get("force_rerun") or not kwargs.get("run_dir"):
            return float("nan")
        try:
            folder = _run_cards_dir(kwargs["run_dir"])
            return str([(i, os.path.getmtime(p)) for i, p in _card_files(folder)])
        except Exception:
            return float("nan")

    def load_cards(self, storyboard_data, run_dir, require_complete=True,
                   force_rerun=False):
        from PIL import Image

        data = json.loads(storyboard_data)
        segments = data.get("segments", [])
        required = data.get("card_count", len(segments) + 1)
        folder = _run_cards_dir(run_dir)
        entries = _card_files(folder)

        usable = [(i, p) for i, p in entries if i < required]
        stale = [i for i, _ in entries if i >= required]

        lines = ["=" * 78,
                 " STORYBOARD DECK",
                 "=" * 78,
                 f" Folder   : {folder}",
                 f" Rendered : {len(usable)}/{required} cards",
                 f" Segments : {len(segments)} video clips (card i -> card i+1)"]
        if stale:
            lines.append(f" NOTE     : ignoring {len(stale)} leftover card(s) {stale[:6]} from a longer run."
                         " Change 'project_name' for a fresh deck.")

        have_indices = {idx for idx, _ in usable}
        missing = [i for i in range(required) if i not in have_indices]
        if missing:
            lines.append(f" MISSING  : {missing[:12]}{' ...' if len(missing) > 12 else ''}")

        if not usable:
            lines += ["-" * 78,
                      " No cards rendered yet.",
                      " Set the Queue batch count to the card total above and queue once -",
                      " 'card_index' auto-increments so every card renders in turn.",
                      "=" * 78]
            report = "\n".join(lines)
            _safe_print(f"[StoryboardCardLoader]\n{report}")
            return {"ui": {"text": (report,)}, "result": (_block(), 0, report)}

        # Every card must share one size to form a batch; the first card sets it.
        first = Image.open(usable[0][1]).convert("RGB")
        width, height = first.size
        frames = []
        for _index, path in usable:
            image = Image.open(path).convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.LANCZOS)
            frames.append(np.asarray(image, dtype=np.float32) / 255.0)

        batch = torch.from_numpy(np.stack(frames))
        lines += ["-" * 78, f" Loaded batch: {tuple(batch.shape)}", "=" * 78]
        report = "\n".join(lines)
        _safe_print(f"[StoryboardCardLoader]\n{report}")

        if require_complete and missing:
            return {"ui": {"text": (report,)}, "result": (_block(), len(usable), report)}
        return {"ui": {"text": (report,)}, "result": (batch, len(usable), report)}


def _win_commit():
    """(free, total) Windows commit memory in bytes - RAM plus page file.

    This is the number that actually decides whether the next model can be
    staged, and NOTHING in ComfyUI's log reports it. When dynamic VRAM streams
    weights it pins host buffers sized at twice the model, charged against
    commit; when commit runs out the failure is not a clean OOM - the host read
    fails ("HostBuffer.read_file_slice failed"), CUDA then reports out of
    memory, and the prompt worker thread dies with the GPU still allocated.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return (status.ullAvailPageFile, status.ullTotalPageFile,
                status.ullAvailPhys, status.ullTotalPhys)
    except Exception:
        return None


GB = float(1 << 30)


def _memory_line():
    """One line of the three numbers that matter, for the console and panels."""
    parts = []
    try:
        if torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info()
            parts.append(f"VRAM {free_vram / GB:5.1f} / {total_vram / GB:.1f} GB free")
    except Exception:
        pass
    commit = _win_commit()
    if commit:
        free_commit, total_commit, free_ram, _total_ram = commit
        parts.append(f"commit {free_commit / GB:5.1f} / {total_commit / GB:.1f} GB free")
        parts.append(f"RAM {free_ram / GB:5.1f} GB free")
    return "  |  ".join(parts)


def _memory_warning():
    """Loud, specific warning while there is still a chance to act on it."""
    commit = _win_commit()
    if not commit:
        return ""
    free_commit = commit[0] / GB
    if free_commit >= 16.0:
        return ""
    return (
        "\n" + "!" * 78 +
        f"\n LOW COMMIT MEMORY: only {free_commit:.1f} GB of RAM+page-file left."
        "\n Staging a big model needs about TWICE its size in pinned host memory,"
        "\n so this run is likely to die with 'HostBuffer.read_file_slice failed'"
        "\n followed by a CUDA out-of-memory that kills the prompt worker thread."
        "\n FIX: restart ComfyUI with --disable-dynamic-vram"
        "\n      (run_nvidia_gpu_stable_memory.bat), and/or raise the page file."
        "\n" + "!" * 78)


def _stage_cleanup(tag):
    """
    Unload every model before a stage that loads its own big one. The render
    nodes call this themselves so the cleanup happens even in a hand-built graph
    with no StageMemoryCleaner node; ComfyUI reloads what the stage needs.
    """
    before = _memory_line()
    if mm is not None:
        try:
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception as exc:
            _safe_print(f"[{tag}] Warning during model unload: {exc}")
        # Deliberately NOT calling mm.reset_cast_buffers() here. ComfyUI already
        # calls it in a `finally` after EVERY node, so an extra call buys
        # nothing - and it is not free: for each dynamic model it REBUILDS the
        # pinned host buffer, which pinned_hostbuf_size() sizes at twice the
        # model. Requesting a fresh multi-GB pinned allocation immediately
        # before a stage stages a 14 GB text encoder is how a tight run turns
        # into a dead worker thread. Unloading models is what frees memory.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    _safe_print(f"[{tag}] Cleared models before rendering."
                f"\n[{tag}]   before: {before}"
                f"\n[{tag}]   after : {_memory_line()}")
    warning = _memory_warning()
    if warning:
        _safe_print(warning)


class StageMemoryCleaner:
    """
    Explicitly unloads all active AI models from GPU VRAM (e.g. after Audio Extraction or Image Gen)
    to guarantee zero VRAM overhang before starting the next generation stage (e.g. LTX 2.5 Video Gen).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough_data": (ANY, ),
                "stage_name": ("STRING", {"default": "Stage Transition: Clear VRAM"}),
            }
        }

    RETURN_TYPES = (ANY, )
    RETURN_NAMES = ("PASSTHROUGH", )
    FUNCTION = "clean_memory"
    CATEGORY = "Geekatplay/VideoClipMaker"

    def clean_memory(self, passthrough_data, stage_name="Stage Transition: Clear VRAM"):
        _stage_cleanup(f"StageMemoryCleaner: {stage_name}")
        return (passthrough_data, )


class AudioLyricsSegmenter(ForceRerunMixin):
    """
    Extracts/aligns lyrics with song timeline and dynamically splits audio into 5-10 sec video segments.
    Calculates exact dynamic card count (N segments -> N+1 keyframe cards) based on total audio duration.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_rerun": FORCE_RERUN_INPUT,
                "target_duration_sec": ("FLOAT", {"default": 6.0, "min": MIN_SEGMENT_SEC, "max": MAX_SEGMENT_SEC, "step": 0.5}),
                "whisper_mode": (["none (auto-split by duration & custom lyrics)",
                                  "tiny", "base", "small", "medium", "large-v3"],
                                 {"default": "large-v3"}),
            },
            "optional": {
                "audio": ("AUDIO", ),
                "audio_path": ("STRING", {
                    "default": "",
                    "placeholder": r"Browse to any song on disk, e.g. D:\Music\my_track.mp3 — or leave empty and use the connected Load Audio node",
                }),
                "custom_lyrics": ("STRING", {"multiline": True, "default": "Verse 1:\nWalking through the neon lights\nDreaming of the stars tonight\n\nChorus:\nWe run until the break of day\nLost in music far away"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("STORYBOARD_DATA", "SEGMENT_COUNT", "AUDIO_PATH")
    FUNCTION = "segment_audio"
    CATEGORY = "Geekatplay/VideoClipMaker"

    def segment_audio(self, target_duration_sec, whisper_mode, audio=None, audio_path="", custom_lyrics="", force_rerun=False):
        typed_path = _resolve_audio_path(audio_path)

        if typed_path:
            resolved_path = typed_path
            total_duration = _probe_duration(resolved_path)
            source = f"path: {resolved_path}"
            if audio is not None:
                print("[AudioLyricsSegmenter] Both a file path and a Load Audio input are present - "
                      "the typed path wins. Clear 'audio_path' to use the connected node instead.")
        elif audio is not None:
            # Exact duration straight from the decoded waveform - no ffprobe guesswork.
            total_duration = _duration_from_waveform(audio)
            resolved_path = _write_temp_wav(audio)
            source = "connected Load Audio node"
        else:
            raise ValueError(
                "AudioLyricsSegmenter has no song. Either connect a 'Load Audio' node to the "
                "'audio' input (it has a file browser with upload and preview), or type/paste the "
                "full path of a song file into 'audio_path'."
            )

        if total_duration <= 0.0:
            raise ValueError(f"Could not determine a usable duration for the song ({source}).")

        _safe_print(f"[AudioLyricsSegmenter] Song source -> {source}  |  duration {total_duration:.2f}s")

        music = _analyse_music(resolved_path)
        if music:
            _safe_print(f"[AudioLyricsSegmenter] Music: {music['tempo']:.0f} BPM, "
                        f"{music['key']} {music['mode']}, sounds like "
                        f"'{music['look'].replace('_', ' ')}' -> suggests "
                        f"'{music['suggested_preset']}'.")
        audio_path = resolved_path

        # Dynamically calculate number of segments and card count for song length.
        # An even split keeps every segment identical and inside the 5-10s rule.
        num_segments = max(1, int(np.ceil(total_duration / target_duration_sec)))
        segment_duration = total_duration / num_segments

        # CUT ON THE BEAT. Each interior boundary moves to the nearest detected
        # beat, as long as both neighbouring segments stay inside the 5-10s
        # rule. Every cut between clips then lands exactly on a beat of the
        # song, which is how music videos get their sense of rhythm - the video
        # model itself cannot hear the track.
        boundaries = [i * segment_duration for i in range(1, num_segments)]
        beat_times = (music or {}).get("beat_times") or []
        snapped = 0
        if beat_times:
            edges = [0.0] + list(boundaries) + [total_duration]
            for k in range(1, len(edges) - 1):
                target = edges[k]
                beat = min(beat_times, key=lambda t: abs(t - target))
                lo = edges[k - 1] + MIN_SEGMENT_SEC
                hi = min(edges[k + 1] - MIN_SEGMENT_SEC,
                         edges[k - 1] + MAX_SEGMENT_SEC)
                if lo <= beat <= hi and abs(beat - target) > 0.02:
                    edges[k] = beat
                    snapped += 1
                elif lo <= beat <= hi:
                    edges[k] = beat
            boundaries = edges[1:-1]
            if snapped:
                _safe_print(f"[AudioLyricsSegmenter] {snapped}/{num_segments - 1} segment "
                            f"cuts snapped onto the beat grid ({music['tempo']:.0f} BPM).")

        edges = [0.0] + list(boundaries) + [total_duration]
        segments = []
        for i in range(num_segments):
            start_t, end_t = edges[i], edges[i + 1]
            segments.append({
                "segment_index": i,
                "start_time": round(start_t, 2),
                "end_time": round(end_t, 2),
                "duration": round(end_t - start_t, 2),
                "on_beat": bool(beat_times) and i > 0,
                "lyrics": INSTRUMENTAL,
            })

        # ---- lyrics: transcribe from the song, or fall back to the typed lyrics ----
        transcript, backend = [], None
        if whisper_mode and not whisper_mode.strip().lower().startswith("none"):
            transcript, backend = _transcribe_song(resolved_path, whisper_mode.strip(),
                                                   lyrics_hint=custom_lyrics)
            if transcript:
                _assign_transcript_to_segments(transcript, segments)
                sung = sum(1 for s in segments if s["lyrics"] != INSTRUMENTAL)
                _safe_print(f"[AudioLyricsSegmenter] {backend}: {len(transcript)} timed lyric tokens "
                            f"-> {sung}/{len(segments)} segments have words.")
            elif backend:
                _safe_print("[AudioLyricsSegmenter] Transcription found no words - the track may be "
                            "purely instrumental. Using 'custom_lyrics' instead.")

        if not transcript:
            lines = _clean_lyric_lines(custom_lyrics.splitlines() if custom_lyrics else [])
            if not lines:
                lines = ["Instrumental section", "Melodic climax", "Rhythmic beat", "Outro sequence"]
            for seg, text in zip(segments, _distribute_lyrics(lines, num_segments)):
                seg["lyrics"] = text
            _safe_print(f"[AudioLyricsSegmenter] Using custom_lyrics: {len(lines)} lines spread "
                        f"across {num_segments} segments.")

        if music:
            _annotate_segments_with_music_structure(segments, music, total_duration)

        storyboard_data = {
            "audio_path": audio_path,
            "total_duration": round(total_duration, 2),
            "target_duration": target_duration_sec,
            "card_count": num_segments + 1,  # N segments require N+1 keyframe cards dynamically
            "lyric_source": whisper_mode if transcript else "custom_lyrics",
            "music": music,
            # Kept so retiming in SongTimelineReview can realign words to the new windows.
            "transcript": transcript,
            "segments": segments
        }

        # Purge transcription model VRAM memory
        if mm is not None:
            mm.soft_empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (json.dumps(storyboard_data, indent=2), len(segments), audio_path)


class SongTimelineReview(ForceRerunMixin):
    """
    STEP 1 PAUSE. Shows how the song was mapped to the timeline and holds the workflow
    here until the mapping is approved. Segment lengths can be retimed before approving.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "approve_timeline": ("BOOLEAN", {
                    "default": False,
                    "label_on": ">>> APPROVED - saves the run, continue",
                    "label_off": "*** PAUSED - NOTHING IS SAVED YET ***",
                }),
                "force_rerun": FORCE_RERUN_INPUT,
                "storyboard_data": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "segment_durations": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Leave empty for the automatic even split.\nOr retime per segment, e.g.  6, 6, 8, 5.5, 10",
                }),
                "style_preset": (list(STYLE_PRESETS.keys()), {"default": "Cinematic film"}),
                "style_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the video in your own words - mood, era, palette, "
                                   "camera, anything. Added to EVERY card prompt on top of the preset.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("STORYBOARD_DATA", "PREVIEW_DATA", "TIMELINE_TABLE", "FULL_LYRICS",
                    "SEGMENT_COUNT")
    FUNCTION = "review_timeline"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = ("Step 1 checkpoint: review the song-to-timeline mapping, optionally retime "
                   "segments, then approve to let the storyboard step run.")

    def review_timeline(self, storyboard_data, approve_timeline, segment_durations="",
                        style_preset="Cinematic film", style_description="", force_rerun=False):
        data = json.loads(storyboard_data)
        # Travels with the timeline into the project file, so every card prompt is
        # built with it without you having to retype it later.
        data["style_preset"] = style_preset
        data["style_description"] = (style_description or "").strip()
        data["global_prompt"] = _compose_style(style_preset, style_description,
                                              data.get("music"))
        total_duration = float(data.get("total_duration", 0.0))
        notes = []

        overrides = _parse_durations(segment_durations)
        if overrides:
            flat_lines = []
            for seg in data.get("segments", []):
                if seg["lyrics"] != INSTRUMENTAL:
                    flat_lines.extend(p.strip() for p in seg["lyrics"].split(" / ") if p.strip())

            segments, notes = _build_segments(
                total_duration, overrides, flat_lines, float(data.get("target_duration", 6.0))
            )

            # With a timed transcript we can realign the actual words to the new windows
            # instead of reshuffling text that was already bucketed by the old timing.
            transcript = data.get("transcript") or []
            if transcript:
                _assign_transcript_to_segments(transcript, segments)
                notes.append(f"lyrics realigned to the new windows from {len(transcript)} timed tokens")

            data["segments"] = segments
            data["card_count"] = len(segments) + 1
            data["retimed"] = True

        segments = data.get("segments", [])
        out_of_range = [s["segment_index"] for s in segments
                        if not (MIN_SEGMENT_SEC - 0.01 <= s["duration"] <= MAX_SEGMENT_SEC + 0.01)]
        if out_of_range:
            notes.append(f"segments marked '!' are outside the {MIN_SEGMENT_SEC:g}-{MAX_SEGMENT_SEC:g}s rule: {out_of_range}")

        # Rides along to StoryboardProjectSave, which writes a rewritable draft run
        # while this is false and a permanent timestamped run once it is true.
        data["approved"] = bool(approve_timeline)

        if approve_timeline:
            hints = [
                " Timeline APPROVED. The storyboard prompts for "
                f"{len(segments) + 1} cards are being generated.",
                " This queue is saved as its own run, marked approved.",
                " Turn approve_timeline off again to come back and retime.",
            ]
        else:
            hints = [
                " HOW TO RETIME : type per-segment seconds into 'segment_durations',",
                "                 e.g.  6, 6, 8, 5.5, 10   (anything left off keeps the auto length)",
                " This queue IS saved - Part 2 can open it right away. Every queue writes",
                " its own run folder, so nothing you have already made is overwritten.",
                " Flip the TOP toggle 'approve_timeline' to APPROVED once the timing is",
                " right, to mark which run you settled on.",
            ]

        table = _timeline_table(data, "STEP 1 - SONG TIMELINE MAPPING", approve_timeline, notes, hints)
        print(f"[SongTimelineReview]\n{table}")

        if not approve_timeline:
            # The table and the lyric sheet always pass through: you must be able to
            # read what was transcribed while the pipeline is still paused.
            # PREVIEW_DATA always flows: you must be able to see the card prompts
            # for the current timeline and style before deciding to approve.
            return {"ui": {"text": (table,)},
                    "result": (_block(), json.dumps(data, indent=2), table,
                               _lyrics_sheet(data), _block())}

        payload = json.dumps(data, indent=2)
        return {"ui": {"text": (table,)},
                "result": (payload, payload, table, _lyrics_sheet(data), len(segments))}


class StoryboardCardReview(ForceRerunMixin):
    """
    STEP 2 PAUSE. Holds the workflow after the storyboard card images are rendered so they
    can be inspected (and single cards rerendered) before any video model is loaded.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "approve_cards": ("BOOLEAN", {
                    "default": False,
                    "label_on": ">>> APPROVED - go make the video",
                    "label_off": "*** PAUSED - reviewing the cards ***",
                }),
                "force_rerun": FORCE_RERUN_INPUT,
                "images": ("IMAGE", ),
                "storyboard_data": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("CARD_IMAGES", "CARD_REPORT")
    FUNCTION = "review_cards"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = ("Step 2 checkpoint: verify every storyboard card rendered, then approve to "
                   "start LTX 2.5 video generation.")

    def review_cards(self, images, storyboard_data, approve_cards, force_rerun=False):
        data = json.loads(storyboard_data)
        segments = data.get("segments", [])
        required = data.get("card_count", len(segments) + 1)
        rendered = int(images.shape[0])

        bar = "=" * 78
        status = "APPROVED - video generation starts" if approve_cards else "NOT APPROVED - workflow paused here"
        rows = [
            bar,
            f" STEP 2 - STORYBOARD CARD REVIEW   [ {status} ]",
            bar,
            f" Cards rendered : {rendered}",
            f" Cards required : {required}   ({len(segments)} segments + 1 closing frame)",
            f" Video segments : {len(segments)}  (each pairs card i -> card i+1)",
            "-" * 78,
        ]

        if rendered < required:
            rows.append(f" WARNING: {required - rendered} card(s) missing. The last card would be reused,")
            rows.append("          so those segments will not move. Render the rest before approving.")
        elif rendered > required:
            rows.append(f" NOTE: {rendered - required} extra card(s) in the batch; the surplus is ignored.")
        else:
            rows.append(" STATUS: card count matches the timeline exactly.")

        rows.append("-" * 78)
        if approve_cards:
            rows.append(" Cards approved. Image models are unloaded and LTX 2.5 takes over.")
        else:
            rows.extend([
                " REVIEW  : look through the rendered cards above/left.",
                " REDO ONE: set 'card_index' on Storyboard Card Selector, type a new prompt into",
                "           'override_prompt', and rerender just that card.",
                " WHEN OK : flip 'approve_cards' on and queue again to start video generation.",
            ])
        rows.append(bar)
        report = "\n".join(rows)
        print(f"[StoryboardCardReview]\n{report}")

        if not approve_cards:
            return {"ui": {"text": (report,)}, "result": (_block(), report)}

        return {"ui": {"text": (report,)}, "result": (images, report)}


class StoryboardPromptGenerator(ForceRerunMixin):
    """
    Generates prompts for all N+1 keyframe images (Cards 0..N) and dynamic motion prompts for video generation.
    Driven by music analysis (tempo, energy, percussive beat, dynamics) and subject mode.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_rerun": FORCE_RERUN_INPUT,
                "storyboard_data": ("STRING", {"forceInput": True}),
                "subject_mode": (SUBJECT_MODES, {
                    "default": SUBJECT_MODES[0],
                    "tooltip": "Select who or what appears in the video: Auto (smart detection), "
                               "Singer/Performer, Single Person, Band/Group, Crowd, or No People (pure scenery)."
                }),
                "subject": ("STRING", {"multiline": True, "default": "",
                                       "placeholder": "Leave EMPTY to derive the character from the "
                                                      "lyrics. Or describe your own.",
                                       "tooltip": "The recurring character/subject. Empty = read the "
                                                  "song and decide who it is about."}),
                "setting": ("STRING", {"multiline": True, "default": "",
                                       "placeholder": "Leave EMPTY to derive the world from the "
                                                      "lyrics. Or describe your own.",
                                       "tooltip": "The world every shot lives in. Empty = read the "
                                                  "song and decide where it happens."}),
                "visual_style": ("STRING", {"multiline": True, "default": "",
                                            "placeholder": "Leave empty - the look comes from "
                                                           "Part 1's style preset + the music "
                                                           "analysis. Type here only to add more."}),
                "motion_style": ("STRING", {"multiline": True, "default": "",
                                            "placeholder": "Leave empty - camera and pacing come "
                                                           "from the song's mood and tempo. Type "
                                                           "here only to add more."}),
                "image_generator_model": (["Z-Image", "FLUX.1-Dev", "Qwen-Image", "SDXL", "Custom"], {"default": "Z-Image"}),
                "prompt_approach": (PROMPT_APPROACHES, {
                    "default": PROMPT_APPROACHES[0],
                    "tooltip": "Literal follows each lyric line. Creative reads the whole "
                               "song first - dominant mood, recurring imagery - and lets "
                               "details leak between verses. Abstract paints the mood "
                               "instead of the words.",
                }),
                "global_prompt_override": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Empty = use the global style set in Part 1. Type here to override it.",
                }),
                "vary_shots": ("BOOLEAN", {"default": True, "label_on": "vary camera per card", "label_off": "same framing"}),
                "prompt_writer": (PROMPT_WRITERS, {
                    "default": PROMPT_WRITERS[0],
                    "tooltip": "Rule tables: instant, offline, but the imagery comes from "
                               "fixed keyword lists and the framing cycles by card number. "
                               "Local LLM: reads the whole song once and writes every card "
                               "against that reading, so the deck follows THIS song. Needs "
                               "an instruct model and about a minute before Part 2 starts. "
                               "Falls back to the rule tables if the model cannot be loaded.",
                }),
            },
            "optional": {
                "llm_model": ("STRING", {
                    "default": "Qwen/Qwen3-4B-Instruct-2507",
                    "placeholder": "folder under ComfyUI/models/LLM, or a Hugging Face repo id",
                    "tooltip": "A folder name under ComfyUI/models/LLM wins over a repo id, so "
                               "an offline install never reaches for the network. Any instruct "
                               "('-Instruct' / '-it') checkpoint works; 4B is enough, and a "
                               "Gemma 3 or Qwen-VL you already have for LTX will do.",
                }),
                "llm_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                     "control_after_generate": True,
                                     "tooltip": "Same song + same seed = the same storyboard. "
                                                "Change it to get a different reading of the "
                                                "same song."}),
                "llm_temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.5, "step": 0.05,
                                              "tooltip": "How far the model roams. Below ~0.5 it "
                                                         "repeats itself across cards; above ~1.1 "
                                                         "it starts ignoring the lyrics."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("PROMPT_LIST", "ANIMATION_PROMPTS", "REFERENCE_PLAN", "CARD_SHEET",
                    "CARD_COUNT")
    FUNCTION = "generate_prompts"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"

    def generate_prompts(self, storyboard_data, subject, setting, visual_style, motion_style,
                         image_generator_model, prompt_approach=PROMPT_APPROACHES[0],
                         global_prompt_override="", vary_shots=True,
                         subject_mode=SUBJECT_MODES[0], force_rerun=False,
                         prompt_writer=PROMPT_WRITERS[0], llm_model="",
                         llm_seed=0, llm_temperature=0.85):
        data = json.loads(storyboard_data)
        segments = data.get("segments", [])
        card_count = data.get("card_count", len(segments) + 1)
        music = data.get("music") or {}
        quality = MODEL_QUALITY_TOKENS.get(image_generator_model, "")
        global_prompt = (global_prompt_override or "").strip() or (data.get("global_prompt") or "").strip()

        theme = _song_theme(segments)
        card_prompts = animation_prompts = references = ref_assignment = story = None
        writer_used = "rule tables"

        # The local model reads the song and decides WHO, WHERE and every shot
        # in one go - subject, setting and references included, because they
        # only stay consistent if one reading produces all of them.
        if str(prompt_writer).startswith("local LLM"):
            try:
                from .llm_writer import write_storyboard
                card_prompts, animation_prompts, plan, reading = write_storyboard(
                    storyboard_data, llm_model, seed=llm_seed,
                    temperature=llm_temperature, approach=prompt_approach)
                references, ref_assignment = plan["references"], plan["assignment"]
                subject, setting = reading["subject"], reading["setting"]
                subject_source = setting_source = f"written by {llm_model}"
                writer_used = f"local LLM ({llm_model})"
            except Exception as exc:
                # A missing checkpoint, an out-of-memory card or a reply that
                # will not parse must never take the queue down with it.
                _safe_print("[StoryboardPromptGenerator] " + "!" * 60)
                _safe_print(f"[StoryboardPromptGenerator] Local LLM writer failed: {exc}")
                _safe_print("[StoryboardPromptGenerator] Falling back to the rule tables for "
                            "this run - the storyboard is still usable, just less specific "
                            "to this song.")
                _safe_print("[StoryboardPromptGenerator] " + "!" * 60)
                card_prompts = animation_prompts = references = ref_assignment = None

        concert = str(prompt_approach).startswith("Concert")
        if card_prompts is None:
            # A concert IS a singer on a stage - the approach decides both.
            if concert:
                subject_mode = "Singer / Performer"

            # WHO and WHERE come from the song itself and subject_mode unless typed.
            derived_subject, derived_setting = _derive_subject_setting(segments, music=music, subject_mode=subject_mode)
            subject_source = setting_source = "typed by you"
            if not (subject or "").strip():
                subject = derived_subject
                subject_source = f"derived ({subject_mode})"
                if concert:
                    subject += ", with a signature haircut and one stage outfit"
            if not (setting or "").strip():
                if concert:
                    setting = _concert_stage(music)
                    setting_source = "concert stage (from music look)"
                else:
                    setting = derived_setting or "a place that matches the song's mood"
                    setting_source = "derived from lyrics/music"

            # One reading of the song as a STORY: a journey through locations,
            # a light arc, and a progressing action per segment. Seeded by the
            # lyrics + llm_seed, so the same song retells the same story until
            # you change the seed.
            lyric_key = zlib.crc32(" ".join(s.get("lyrics", "") for s in segments)
                                   .encode("utf-8"))
            rng = random.Random(int(llm_seed or 0) ^ lyric_key)
            if concert:
                story = _compose_concert_story(segments, music, subject, setting, rng)
            else:
                story = _compose_story(segments, theme, music, subject, setting, rng)
            if story:
                for seg, beat in zip(segments, story["beats"]):
                    seg["_story"] = beat

            references, ref_assignment = _plan_references(
                subject, setting, segments, global_prompt=global_prompt,
                subject_mode=subject_mode, story=story
            )

            card_prompts = []
            for i in range(card_count):
                is_finale = i >= len(segments)
                seg = segments[-1] if is_finale else segments[i]
                card_prompts.append(
                    _build_card_prompt(i, card_count, seg, visual_style, quality, is_finale,
                                       vary_shots, subject, setting, image_generator_model,
                                       global_prompt, music=music, subject_mode=subject_mode,
                                       approach=prompt_approach, theme=theme)
                )

            video_style = _video_style(global_prompt)
            animation_prompts = []
            for seg in segments:
                idx = seg["segment_index"]
                animation_prompts.append(
                    _build_motion_prompt(idx, seg, subject, setting, motion_style, video_style,
                                         next_shot=_shot_recipe_smart(idx + 1, music=music)["shot"],
                                         approach=prompt_approach, theme=theme,
                                         music=music, subject_mode=subject_mode)
                )

        reference_plan = {"references": references, "assignment": ref_assignment}

        _safe_print(f"[StoryboardPromptGenerator] Subject Mode: {subject_mode}")
        _safe_print(f"[StoryboardPromptGenerator] Subject ({subject_source}): {subject}")
        _safe_print(f"[StoryboardPromptGenerator] Setting ({setting_source}): {setting}")
        _safe_print(f"[StoryboardPromptGenerator] Built {len(card_prompts)} card prompts and "
                    f"{len(animation_prompts)} motion prompts via {writer_used} "
                    f"(model preset: {image_generator_model}).")
        # An abstract frame has no character, place or object to match, so the
        # reference suffixes would only pull it back toward a literal image.
        skip_refs = (prompt_approach.startswith("Abstract")
                     and writer_used.startswith("rule tables"))
        for i in range(card_count if not skip_refs else 0):
            rid = ref_assignment.get(str(i), "character")
            ref = next((r for r in references if r["id"] == rid), None)
            if ref and ref["kind"] == "scene":
                card_prompts[i] += f" The location is the established {ref['name']} reference."
            elif ref and ref["kind"] == "prop":
                card_prompts[i] += (f" The {ref['name']} is the established object reference, "
                                    f"identical in shape, colour and wear.")
            elif ref and ref["kind"] == "character":
                card_prompts[i] += f" The subject is the established {ref['name']} reference."

        # SDXL's CLIP encoder reads only the first 77 tokens (~300 characters) -
        # everything past that is silently ignored, so a 900-character card
        # prompt renders from its first third and the rest is dead weight.
        # Prose models (Z-Image / FLUX / Qwen) have LLM encoders and keep the
        # full description.
        if image_generator_model not in PROSE_MODELS:
            clipped = 0
            for i, p in enumerate(card_prompts):
                fitted = _fit_prompt(p, 300, keep_tail=0)
                if len(fitted) < len(p):
                    clipped += 1
                card_prompts[i] = fitted
            if clipped:
                _safe_print(f"[StoryboardPromptGenerator] {clipped} card prompt(s) fitted to "
                            f"{image_generator_model}'s 77-token CLIP window (~300 chars); "
                            "the front of each prompt carries the shot, subject and action, "
                            "so that is what was kept.")

        story_text = _story_sheet(story, segments)
        sheet = ((story_text + "\n\n" if story_text else "")
                 + _reference_sheet(references, ref_assignment, card_count) + "\n\n"
                 + _card_sheet(segments, card_count, card_prompts,
                            [f"global style applied to every card: "
                             f"{global_prompt or '(none)'}",
                             f"subject mode: {subject_mode}",
                             f"model preset: {image_generator_model}",
                             f"prompts written by: {writer_used}",
                             f"prompt approach: {prompt_approach}",
                             f"subject ({subject_source}): {subject}",
                             f"setting ({setting_source}): {setting}",
                             f"song theme: {theme['mood'] or 'neutral'}"
                             + (f"; recurring imagery: {', '.join(theme['imagery'][:3])}"
                                if theme['imagery'] else ""),
                             "prose prompts (Z-Image / FLUX / Qwen)" if image_generator_model in PROSE_MODELS
                             else "tag prompts (SDXL style)"]))
        return {"ui": {"text": (sheet,)},
                "result": (json.dumps(card_prompts, indent=2),
                           json.dumps(animation_prompts, indent=2),
                           json.dumps(reference_plan, indent=2), sheet, card_count)}



SHOT_TYPES = [
    "extreme wide establishing shot", "wide shot", "medium wide shot", "medium shot",
    "medium close-up", "close-up", "extreme close-up", "over-the-shoulder shot",
    "low angle hero shot", "high angle looking down", "dutch angle shot",
    "aerial drone shot", "tracking shot from the side", "silhouette backlit shot",
]

LIGHTING = [
    "hard rim lighting with deep shadows", "soft diffused light, gentle falloff",
    "harsh top light, dramatic contrast", "warm golden hour backlight",
    "cold blue moonlight", "neon practicals reflecting on wet surfaces",
    "volumetric god rays through haze", "single hard key light, rest in darkness",
    "bounced fill light, cinematic mid-tones", "flickering coloured light",
]

# Cinematography only - no weather, no season, no scene content. Rain, embers
# and petals used to live in this list and were cycled into every song blind;
# what falls from the sky must come from the lyrics, not from a rotation.
ATMOSPHERE = [
    "drifting atmospheric haze", "floating dust motes in the light",
    "low soft ground mist", "gentle volumetric light",
    "soft bokeh particles", "fine airborne particles catching the light",
    "a subtle depth haze between the layers", "clean crisp air, deep visibility",
]

COMPOSITION = [
    "rule of thirds composition, subject offset left",
    "centered symmetrical composition",
    "strong leading lines toward the subject",
    "deep foreground framing, subject beyond",
    "negative space above the subject",
    "tight framing, shallow depth of field",
]


def _shot_recipe(index):
    """Deterministic but varied cinematography per card, so consecutive cards
    covering the same lyric still look like different shots of one film."""
    return {
        "shot": SHOT_TYPES[index % len(SHOT_TYPES)],
        "light": LIGHTING[(index * 3 + 1) % len(LIGHTING)],
        "atmos": ATMOSPHERE[(index * 5 + 2) % len(ATMOSPHERE)],
        "comp": COMPOSITION[(index * 7 + 3) % len(COMPOSITION)],
    }


# ---------------------------------------------------------------------------
# Motion prompts.
#
# A video model is not an image model. Words that are correct in a card prompt
# ("cinematic film still", "photograph", "poster") tell LTX 2.5 to hold a static
# frame, and a first/last-frame render with nothing else to go on degrades into
# a crossfade between the two cards. The motion prompt therefore drops all
# still-photography language and leads with the camera move and what physically
# moves in the shot.
# ---------------------------------------------------------------------------

STILL_WORDS = re.compile(
    r"\b(cinematic\s+film\s+still|film\s+still|still\s+frame|freeze\s+frame|"
    r"photograph|photorealistic\s+photo|photo|poster|print|artwork|illustration)\b",
    re.IGNORECASE)

# Wide -> tight, so the difference between two cards names a real camera move.
SHOT_TIGHTNESS = (
    ("extreme wide", 0), ("aerial", 0), ("establishing", 0), ("silhouette", 1),
    ("high angle", 1), ("medium wide", 2), ("wide", 1), ("tracking", 2),
    ("low angle", 2), ("dutch", 3), ("over-the-shoulder", 4),
    ("extreme close", 6), ("medium close", 4), ("close", 5), ("medium", 3),
)

SUBJECT_MOTION = [
    "walks slowly forward, clothes and hair moving with each step",
    "turns toward the camera, weight shifting from one foot to the other",
    "raises a hand and tilts their head up, chest rising as they breathe",
    "looks off-frame and then back, eyes tracking something past the lens",
    "steps through the frame from left to right, shoulders swaying",
    "leans in closer, lips parting slightly, blinking once",
    "reaches out toward the light, fingers opening",
    "pushes forward against the wind, head lowered then lifting",
]


def _shot_tightness(shot):
    s = (shot or "").lower()
    for key, rank in SHOT_TIGHTNESS:
        if key in s:
            return rank
    return 3


def _an(phrase):
    return ("an " if (phrase or "")[:1].lower() in "aeiou" else "a ") + phrase


# How the camera moves is a storytelling choice, not a geometry problem. Deriving
# it only from how much the framing tightens made almost every segment "the
# camera dollies in" - a music video that only ever zooms. These pools move the
# camera the way the music FEELS, and cycle so no two neighbouring segments
# repeat the same move.
CAMERA_MOODS = {
    "tender and warm": [
        "the camera drifts in a slow gentle arc around the subject",
        "a soft handheld sway, the frame breathing with the music",
        "the camera glides sideways at walking pace, staying close",
        "a slow tender rise, tilting down as it settles",
    ],
    "desolate and aching": [
        "the camera holds almost still, only the world drifting past in parallax",
        "a slow pull-back, the empty space growing around the subject",
        "a high slow crane down, like something descending to look",
        "a distant lateral track, watching from across the space",
    ],
    "violent and urgent": [
        "a hard whip into a fast tracking shot, the background tearing past",
        "aggressive handheld pursuit, the frame shaking with the impact",
        "a fast lateral dolly that overshoots and drags back",
        "a crash move toward the subject that slams to a hold",
    ],
    "euphoric and soaring": [
        "the camera sweeps upward in a rising crane, horizon dropping away",
        "a wide sweeping orbit, the world wheeling around the subject",
        "the camera flies forward low and fast, then lifts",
        "a spiralling climb, tilting toward the sky",
    ],
    "hazy and weightless": [
        "the camera floats untethered, drifting as if underwater",
        "a slow weightless orbit, edges of the frame softening",
        "a suspended hold, layers of the scene sliding at different speeds",
        "the camera sinks gently, like something settling through deep water",
    ],
    "": [
        "the camera tracks sideways in a slow arc, parallax layers sliding past",
        "a slow orbit around the subject, background wheeling behind",
        "the camera cranes down from above, settling to eye level",
        "a measured pull-back reveal, the space opening up",
        "a lateral dolly keeping pace with the motion",
        "a near-static hold, the environment doing all the moving",
        "the camera drifts through the environment, passing behind foreground "
        "objects that briefly eclipse the subject",
        "the camera moves away from the subject, revealing another part of the scene",
        "a slow perspective change, the camera rotating to see the scene from a new side",
        "the camera travels from a foreground detail back to the wider scene",
        "the camera follows a moving element through the scene, the subject passing in and out of frame",
        "a deliberate move toward the subject",
    ],
}


def _camera_move(shot_a, shot_b, idx=0, mood=""):
    """A camera move chosen by how the music feels, framed by the two cards.

    The mood pool sets the character of the move; every third segment draws from
    the neutral pool instead, so even a single-mood song gets reveals, drifts
    through the environment and moves AWAY from the subject - not one move
    repeated with different adjectives.
    """
    pool = CAMERA_MOODS.get(mood) or CAMERA_MOODS[""]
    if idx % 3 == 2:
        pool = CAMERA_MOODS[""]
    # A different stride than the shot cycle, so move and framing don't lock step.
    move = pool[(idx * 3 + 1) % len(pool)]
    return f"{move}, the framing finding its way from {_an(shot_a)} to {_an(shot_b)}"


def _video_style(style):
    """The project style, with still-image language removed."""
    cleaned = STILL_WORDS.sub("", style or "")
    cleaned = re.sub(r"\s*,\s*,", ",", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .,")
    return cleaned


# ---------------------------------------------------------------------------
# Prompt approaches.
#
# A literal prompt per segment reads each lyric line in isolation, which makes
# every prompt the same shape and misses what the SONG is about. The creative
# approaches first read the whole song - dominant mood, recurring imagery - and
# then let each segment borrow from that pool, so details from other verses
# "leak" into the shot and the prompts stop being one repeated template.
# ---------------------------------------------------------------------------

# mood -> how the abstract mode paints it (colour, form, motion energy)
ABSTRACT_MOODS = {
    "tender and warm": ("warm amber and rose light blooming softly",
                        "curved organic forms leaning into each other",
                        "slow breathing motion, everything drifting closer"),
    "desolate and aching": ("cold desaturated blue-grey with one dying ember of colour",
                            "a single small dark form dwarfed by vast empty negative space",
                            "almost still, a single element slowly falling"),
    "violent and urgent": ("hard red and black slashed with white",
                           "sharp diagonal shapes colliding and shattering",
                           "fast aggressive motion, debris and sparks whipping through"),
    "euphoric and soaring": ("radiant gold and cyan bursting outward",
                             "rising spirals and open sky, horizon dropping away",
                             "accelerating upward motion, light streaking past"),
    "hazy and weightless": ("pale iridescent pastels dissolving into fog",
                            "translucent floating layers, edges melting",
                            "suspended slow-motion drift, dust hanging in light"),
    "": ("deep cinematic colour, one strong accent hue",
         "bold silhouettes against layered depth",
         "a deliberate continuous camera drift"),
}

CREATIVE_ACTIONS = [
    "moves through the scene as if pulled by the music, each gesture landing on the beat",
    "stands almost still while the whole world moves around them",
    "turns away from the light, shadow sweeping across their face",
    "reaches toward something just out of frame that the song keeps promising",
    "walks against a current of drifting debris and glowing particles",
    "closes their eyes, and the scene shifts around them like a memory",
    "is caught mid-motion, hair and clothing sculpted by unseen wind",
    "watches their own reflection ripple and not quite keep up",
]

# Cinematic texture for the creative mode - colour shifts, particles, surreal
# elements and environmental changes, cycled so each segment gets its own.
CREATIVE_CINEMA = [
    "the colour palette slowly shifts across the take, one hue bleeding into the next",
    "fine particles hang in the air and swirl in the subject's wake",
    "the environment itself changes subtly - lights waking up, surfaces beginning to glisten",
    "a surreal touch: something in the background moves the way it shouldn't, gently",
    "textures come alive - fabric, hair and film grain all in slow visible motion",
    "the light source itself travels, dragging long shadows across the scene",
    "foreground shapes pass between camera and subject, briefly eclipsing them",
    "reflections and highlights pulse softly in rhythm with the music",
]


def _song_theme(segments):
    """Read the WHOLE song once: dominant mood, recurring imagery, key words."""
    mood_votes, imagery_pool, words = {}, [], {}
    for seg in segments:
        lyric = seg.get("lyrics", "")
        if lyric == INSTRUMENTAL:
            continue
        imgs, mood = _lyric_imagery(lyric, limit=3)
        for im in imgs:
            if im not in imagery_pool:
                imagery_pool.append(im)
        if mood:
            mood_votes[mood] = mood_votes.get(mood, 0) + 1
        for w in set(_lyric_words(lyric)) - LYRIC_STOPWORDS:
            words[w] = words.get(w, 0) + 1
    mood = max(mood_votes, key=mood_votes.get) if mood_votes else ""
    keywords = [w for w, _ in sorted(words.items(), key=lambda kv: -kv[1])[:8]]
    return {"mood": mood, "imagery": imagery_pool, "keywords": keywords}


# ---------------------------------------------------------------------------
# Story engine.
#
# A deck of cards with the same subject in the same setting doing "mid-motion,
# held in the musical moment" eighty times is not a music video - it is one
# photograph taken eighty times. This reads the song ONCE and writes an actual
# story arc: a journey through several locations, light that changes across the
# song, and an action per segment that MOVES the story forward. Each card is
# then a different scene from that story instead of a re-crop of the same shot.
# A seeded RNG adds controlled variety: same song + same seed = same story,
# new seed = a different telling.
# ---------------------------------------------------------------------------

# section_type -> which chapter of the story this segment belongs to
_STORY_STAGES = {"intro": "opening", "verse": "journey", "build_up": "rising",
                 "chorus_climax": "peak", "bridge": "turn", "outro": "resolution"}

# What the subject DOES at each point of the arc. Written as concrete stage
# directions a camera can photograph; {motif} slots take a lyric image.
_STAGE_ACTIONS = {
    "opening": [
        "seen small and far away, entering the scene for the first time",
        "standing at the edge of the space, taking it in before moving",
        "waking or turning toward the light, the story not yet begun",
        "seen from behind, facing the whole world they are about to cross",
    ],
    "journey": [
        "moving through with purpose, eyes fixed on something ahead",
        "pausing at a detail - touching it, reading it, deciding",
        "passing strangers or shapes who do not see them",
        "carrying something they will not put down",
        "looking back the way they came, then continuing",
        "crossing from shadow into light mid-stride",
    ],
    "rising": [
        "beginning to run, the space blurring at the edges",
        "climbing, each step more urgent than the last",
        "pushing through an obstacle - a door, a crowd, a current",
        "shedding something - a coat, a bag, a hesitation - and accelerating",
    ],
    "peak": [
        "arriving - stopped dead at the centre of the space, everything facing them",
        "arms open or fists closed at the highest point, the world wheeling around",
        "face lit fully for the first time, holding the moment",
        "colliding with what they were moving toward, the frame exploding with light",
    ],
    "turn": [
        "alone again in a suddenly quiet space, changed",
        "seeing their reflection or shadow and not recognising it",
        "letting something fall from their hands in slow motion",
        "standing still while everything that was moving drains away",
    ],
    "resolution": [
        "walking away slowly, lighter than they arrived",
        "stopping one last time to look back at everything crossed",
        "setting something down deliberately and leaving it",
        "at rest at last, the space settling around them",
    ],
}

# How the light travels across the whole song - a visual clock the story runs
# on, chosen by the song's key so a dark song moves toward dawn and a bright
# one burns down to dusk. Position in the song indexes into this arc.
_LIGHT_ARCS = {
    "minor": ["deep night, artificial light the only light",
              "the small hours, empty and electric",
              "cold pre-dawn blue, the night thinning",
              "first light breaking hard across everything"],
    "major": ["soft early light, long cool shadows",
              "full open daylight, everything visible",
              "long golden-hour light, warm and directional",
              "dusk settling in, first lamps waking against the dark"],
}


def _compose_story(segments, theme, music, subject, setting, rng):
    """One reading of the song as a story: a journey through locations with a
    light arc and a progressing action per segment. Returns
    {"logline", "locations": [(id, name, desc)], "beats": [per-segment dict]}."""
    n = len(segments)
    if not n:
        return None

    # --- the places the story travels through -----------------------------
    hits = {}
    for seg in segments:
        text = seg.get("lyrics", "")
        if text == INSTRUMENTAL:
            continue
        words = set(_lyric_words(text)) - LYRIC_STOPWORDS
        for sid, roots, _d in SCENE_HINTS:
            ov = len(words & set(roots))
            if ov:
                hits[sid] = hits.get(sid, 0) + ov
    ranked = [sid for sid, _ in sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))][:4]
    # The story's light arc owns the clock, so a location may not carry its own
    # time of day ("open country at golden hour" under "first light breaking").
    def _timeless(desc):
        return re.sub(r"\s*at (golden hour|night|dusk|dawn|midnight)\b", "", desc)
    locations = [(sid, sid.replace("_", " "),
                  _timeless(next(d for s, _r, d in SCENE_HINTS if s == sid)))
                 for sid in ranked]
    # A journey needs at least three distinct places; synthesize the missing
    # ones as different faces of the main setting so the story still travels.
    base = (setting or "").strip() or "the story's main location"
    variants = ["seen from a distance, small against the sky",
                "deep inside it, close and enveloping",
                "its far edge, where it gives way to open space",
                "seen from above, laid out like a map"]
    rng.shuffle(variants)
    vi = 0
    while len(locations) < 3 and vi < len(variants):
        locations.append((f"main_setting_{vi}", f"the main setting - view {vi + 1}",
                          f"{base}, {variants[vi]}"))
        vi += 1

    arc = _LIGHT_ARCS["minor" if (music or {}).get("mode") == "minor" else "major"]
    imagery_pool = list(theme.get("imagery") or [])
    rng.shuffle(imagery_pool)

    # --- one beat per segment ---------------------------------------------
    beats, last_action = [], None
    climax_loc = locations[-1]
    home_loc = locations[0]
    for i, seg in enumerate(segments):
        pos = i / max(1, n - 1)
        stage = _STORY_STAGES.get(seg.get("section_type", "verse"), "journey")
        if stage == "peak":
            loc = climax_loc
        elif stage == "turn":
            loc = home_loc                      # returning changed
        else:
            loc = locations[min(int(pos * len(locations) * 0.999), len(locations) - 1)]
        pool = [a for a in _STAGE_ACTIONS[stage] if a != last_action]
        action = rng.choice(pool)
        last_action = action
        motif = imagery_pool[i % len(imagery_pool)] if imagery_pool and rng.random() < 0.6 else ""
        beats.append({
            "stage": stage,
            "location_id": loc[0], "location_name": loc[1], "location_desc": loc[2],
            "action": action,
            "light": arc[min(int(pos * len(arc) * 0.999), len(arc) - 1)],
            "motif": motif,
        })

    want = ", ".join(theme.get("keywords", [])[:3]) or "what the song keeps returning to"
    logline = (f"{(subject or 'the subject').strip().rstrip('.')} travels from "
               f"{locations[0][1]} to {locations[-1][1]} across the song, "
               f"chasing {want}; the light moves from '{arc[0]}' to '{arc[-1]}'.")
    return {"logline": logline, "locations": locations, "beats": beats}


# --------------------------------------------------------------------------
# Concert approach. A performance video is a story that never leaves the
# stage: the singer, their outfit and haircut stay identical in every card,
# and only the lighting, framing and crowd energy travel with the song.

# The stage the song sounds like it belongs on, per music look.
CONCERT_STAGES = {
    "dark_electronic":
        "a vast electronic concert stage - towering LED walls in teal and magenta, "
        "laser fans cutting the smoke, a silhouetted crowd under artificial glow",
    "driving_rock":
        "a raw rock arena stage - stacked amps and scaffolding rigs, hard white "
        "backlight through haze, sparks, a roaring crowd pressed to the barrier",
    "upbeat_pop":
        "a bright pop arena stage - colour-blocked LED screens, confetti light, "
        "a glossy runway thrust into a cheering crowd",
    "romantic_ballad":
        "an intimate theatre stage - warm amber spotlights, a single microphone "
        "stand, a soft sea of lights held up in the dark",
    "melancholic":
        "a dim small-club stage - one cold spotlight, drifting haze, a quiet "
        "audience half-lost in the shadows",
    "acoustic_intimate":
        "a small acoustic stage - wooden floor, warm practical bulbs and string "
        "lights, the audience close enough to touch",
    "ambient_dreamy":
        "an ethereal stage lost in fog - pale columns of light, slow drifting "
        "haze, the crowd a soft silhouetted sea",
    "cinematic_neutral":
        "a grand concert stage - clean white key light, deep black surround, "
        "a wide silhouetted audience beyond the stage edge",
    "orchestral_classical":
        "a concert hall stage - gilded proscenium and chandeliers, velvet and "
        "warm tungsten light, the orchestra arrayed behind",
    "roots_americana":
        "an outdoor festival stage at dusk - warm string lights over wooden "
        "boards, road cases and worn amps, a field of swaying crowd beyond",
}

# What the singer DOES at each point of the set. Same stage keys as
# _STAGE_ACTIONS so the section mapping is shared.
_CONCERT_ACTIONS = {
    "opening": [
        "standing in darkness at the microphone as the first spotlight finds them",
        "walking out onto the stage, the crowd noise rising to meet them",
        "head bowed at the microphone stand, waiting for the first line",
        "silhouetted against the rig as the intro plays, still and ready",
    ],
    "journey": [
        "singing into the microphone, working the front edge of the stage",
        "carrying the verse along the stage, making eye contact with the crowd",
        "singing with one hand on the microphone stand, the other marking the phrase",
        "moving across the stage mid-verse, the followspot tracking them",
    ],
    "rising": [
        "driving the build-up forward, gestures growing with the music",
        "stepping up onto the monitor wedge, urging the crowd to rise",
        "pulling the microphone from the stand and pacing as the energy climbs",
        "leaning into the crowd as the lights accelerate behind them",
    ],
    "peak": [
        "hitting the chorus full-voice, arms wide, blinders flaring behind them",
        "belting the hook at the stage edge, the whole crowd singing it back",
        "head thrown back on the biggest note, strobes and smoke erupting",
        "commanding centre stage at the climax, every light on them",
    ],
    "turn": [
        "pulling back to a quiet, intimate delivery in a single spotlight",
        "singing the bridge almost in a whisper, eyes closed, the stage dark around them",
        "kneeling at the stage edge for the stripped-back passage",
        "turning away from the crowd for one private line, silhouetted",
    ],
    "resolution": [
        "holding the final note as the lights fade to a single warm glow",
        "lowering the microphone and taking in the crowd as the song ends",
        "standing spent and smiling in the last spotlight, arm raised",
        "walking slowly upstage into the haze as the final chord rings",
    ],
}

# The lighting rig's own arc across the set, replacing the time-of-day arc.
_CONCERT_LIGHT_ARC = [
    "a single cold spotlight in the dark, the rest of the stage barely visible",
    "warm front light rising, the rig waking up colour by colour",
    "full colour stage wash, beams sweeping the crowd in time with the music",
    "blinders and strobes at full intensity, smoke lit hard from behind",
]


def _concert_stage(music):
    return CONCERT_STAGES.get((music or {}).get("look", ""),
                              CONCERT_STAGES["cinematic_neutral"])


def _compose_concert_story(segments, music, subject, setting, rng):
    """The concert as a story: three views of ONE stage, performance actions
    per section, and the light rig as the arc. Same shape as _compose_story
    so references, cards and motion prompts need no special casing."""
    n = len(segments)
    if not n:
        return None

    base = (setting or "").strip() or _concert_stage(music)
    locations = [
        ("stage_wide", "the stage - wide from the crowd",
         f"{base}, seen wide from within the audience"),
        ("stage_front", "the stage - front, at the microphone",
         f"{base}, tight at the front microphone, the crowd behind the lens"),
        ("stage_side", "the stage - from the wings",
         f"{base}, seen from the wings through cross-light and haze"),
    ]

    beats, last_action = [], None
    for i, seg in enumerate(segments):
        pos = i / max(1, n - 1)
        stage = _STORY_STAGES.get(seg.get("section_type", "verse"), "journey")
        loc = locations[1] if stage in ("peak", "rising") else locations[i % len(locations)]
        pool = [a for a in _CONCERT_ACTIONS[stage] if a != last_action] or _CONCERT_ACTIONS[stage]
        action = rng.choice(pool)
        last_action = action
        beats.append({
            "stage": stage,
            "location_id": loc[0], "location_name": loc[1], "location_desc": loc[2],
            "action": action,
            "light": _CONCERT_LIGHT_ARC[min(int(pos * len(_CONCERT_LIGHT_ARC) * 0.999),
                                            len(_CONCERT_LIGHT_ARC) - 1)],
            "motif": "",
        })

    logline = (f"{(subject or 'the singer').strip().rstrip('.')} performs the whole song "
               f"live on one stage - same face, same haircut, same outfit in every shot; "
               f"only the light, the framing and the crowd's energy travel with the song.")
    return {"logline": logline, "locations": locations, "beats": beats}


def _story_sheet(story, segments):
    """The story as a readable synopsis, so you can check it BEFORE rendering."""
    if not story:
        return ""
    rows = ["=" * 78, " STORY - one reading of the song (change llm_seed for another telling)",
            "=" * 78, f" {_wrap(story['logline'], 74, 1)}", "-" * 78]
    for i, (seg, b) in enumerate(zip(segments, story["beats"])):
        rows.append(f" [{i:03d}] {b['stage']:<10} @ {b['location_name']:<22} "
                    f"{_wrap(b['action'], 70, 8)}")
    rows.append("=" * 78)
    return "\n".join(rows)


def _beat_pacing(music):
    """How the movement should ride the tempo. The video model cannot hear the
    song, so its speed of motion has to be written into the prompt."""
    tempo = float((music or {}).get("tempo") or 0)
    if not tempo:
        return ""
    if tempo < 85:
        return (f"The movement is unhurried, long slow phrases of motion riding the "
                f"{tempo:.0f} BPM tempo.")
    if tempo <= 115:
        return (f"Movement keeps an easy pulse with the {tempo:.0f} BPM beat, "
                f"gestures and camera accents landing on the downbeat.")
    return (f"The motion is energetic and rhythmic at {tempo:.0f} BPM - hits, steps "
            f"and direction changes landing on the driving beat.")

def _music_driven_lighting(music, index=0):
    """
    Derive continuous adaptive lighting and atmospheric parameters from music analysis.
    Uses continuous parametric audio signal metrics (energy, percussive ratio, brightness, key mode, contrast).
    """
    if not music:
        return LIGHTING[(index * 3 + 1) % len(LIGHTING)], ATMOSPHERE[(index * 5 + 2) % len(ATMOSPHERE)]

    energy = float(music.get("energy", 0.15))
    percussive = float(music.get("percussive_ratio", 0.3))
    brightness = float(music.get("brightness", 0.25))
    contrast_db = float(music.get("contrast_db", 20.0))
    mode = music.get("mode", "major")

    high_contrast = contrast_db > 22.0 or percussive > 0.38 or energy > 0.2
    warm_palette = mode == "major" and brightness > 0.22

    # contrast_db picks the branch but never appears in the text - an image
    # model cannot use "(22.8 dB)", it is analysis debris in a prompt.
    if high_contrast and warm_palette:
        light = "radiant golden sunflare with high specular contrast, vibrant highlights"
    elif high_contrast:
        light = "hard directional rim lighting with deep shadow contrast and specular sheen"
    elif warm_palette:
        light = "soft golden hour volumetric illumination with warm organic halation"
    else:
        light = "cool diffused atmospheric light with soft volumetric shadow gradients"

    if energy > 0.22:
        atmos = "swirling smoke, whipping wind particles and energetic volumetric light beams"
    elif brightness > 0.3:
        atmos = "glowing airborne dust motes hanging in shafts of crisp clear light"
    elif percussive < 0.2:
        atmos = "dense rolling ground fog and slow floating mist layers"
    else:
        atmos = "subtle atmospheric haze and soft particle drift"

    return light, atmos


def _music_driven_camera(music, idx=0):
    """Dynamic camera motion pool selection based on music tempo and energy."""
    if not music:
        return _camera_move(SHOT_TYPES[idx % len(SHOT_TYPES)], SHOT_TYPES[(idx + 1) % len(SHOT_TYPES)], idx, "")

    tempo = float(music.get("tempo", 100))
    energy = float(music.get("energy", 0.15))
    look = music.get("look", "")

    if tempo >= 120 or energy >= 0.2:
        mood_key = "violent and urgent" if music.get("mode") == "minor" else "euphoric and soaring"
    elif tempo <= 90 or look in ("ambient_dreamy", "romantic_ballad"):
        mood_key = "hazy and weightless" if look == "ambient_dreamy" else "tender and warm"
    elif look == "melancholic":
        mood_key = "desolate and aching"
    else:
        mood_key = ""

    pool = CAMERA_MOODS.get(mood_key) or CAMERA_MOODS[""]
    if idx % 3 == 2:
        pool = CAMERA_MOODS[""]
    move = pool[(idx * 3 + 1) % len(pool)]
    return move


def _music_driven_motion(music, idx, subject, setting, subject_mode):
    """Generate kinetic physical motion tailored to music tempo, beat, and subject mode."""
    tempo = float((music or {}).get("tempo") or 0)
    percussive = float((music or {}).get("percussive_ratio") or 0)
    high_energy = tempo >= 115 or percussive >= 0.38

    if subject == "no_people" or subject_mode == "No People (Scenery & Objects only)":
        env_actions = [
            "atmospheric light beams sweep across architectural surfaces in rhythm",
            "foliage and airborne particles swirl through the frame in a gentle breeze",
            "reflections on wet surfaces ripple softly with ambient light pulses",
            "drifting smoke and mist roll across the ground, revealing hidden depth",
            "shadows slowly elongate and travel across the environment",
            "water droplets glistened and fell in slow-motion parallax",
        ]
        return env_actions[idx % len(env_actions)]

    if subject_mode == "Singer / Performer":
        singer_actions = [
            "sings with intense emotion into a microphone, head tilting back as stage lights flare",
            "gestures expressively to the vocal phrasing, chest rising with each breath",
            "leans forward toward the lens, lips moving in sync with the song's climax",
            "sways with the music rhythm, stage spotlights tracing their movement",
        ]
        return singer_actions[idx % len(singer_actions)]

    if subject_mode == "Band / Group (Multiple People)":
        band_actions = [
            "plays together in tight rhythm, guitarist strumming downbeat while drummer hits cymbal",
            "moves in unison with the music energy, stage lights sweeping across the band",
            "lead singer steps forward while band members perform passionately in the background",
            "band members share a high-energy moment as light bursts behind the stage",
        ]
        return band_actions[idx % len(band_actions)]

    if subject_mode == "Crowd / Atmospheric People":
        crowd_actions = [
            "moves and sways to the music beat, silhouettes illuminated by flashing lights",
            "raises hands in unison as volumetric light cuts through the atmospheric haze",
            "dances through the frame, dynamic motion blur trailing light reflections",
            "gathers around the central light source, energy pulsing through the crowd",
        ]
        return crowd_actions[idx % len(crowd_actions)]

    # Single Person / default
    if high_energy:
        return "moves briskly with kinetic energy, steps and gestures hitting the driving beat"
    return SUBJECT_MOTION[idx % len(SUBJECT_MOTION)]


def _shot_recipe_smart(index, music=None):
    """Dynamic shot recipe incorporating music energy and tempo for framing and lighting."""
    tempo = float((music or {}).get("tempo") or 100)
    energy = float((music or {}).get("energy") or 0.15)
    light, atmos = _music_driven_lighting(music, index)

    if tempo >= 120 or energy >= 0.2:
        shots = ["low angle hero shot", "medium close-up", "tracking shot from the side",
                 "dutch angle shot", "close-up", "extreme close-up"]
        shot = shots[index % len(shots)]
    elif tempo <= 90:
        shots = ["extreme wide establishing shot", "wide shot", "sweeping aerial view",
                 "medium wide shot", "silhouette backlit shot"]
        shot = shots[index % len(shots)]
    else:
        shot = SHOT_TYPES[index % len(SHOT_TYPES)]

    return {
        "shot": shot,
        "light": light,
        "atmos": atmos,
        "comp": COMPOSITION[(index * 7 + 3) % len(COMPOSITION)],
    }


def _build_motion_prompt(idx, seg, who, setting, motion_style, video_style,
                         next_shot=None, approach=PROMPT_APPROACHES[0], theme=None,
                         music=None, subject_mode=SUBJECT_MODES[0]):
    """One segment's video prompt: camera move, physical movement, light, pacing."""
    theme = theme or {"mood": "", "imagery": [], "keywords": []}
    abstract = approach.startswith("Abstract")
    # Concert rides the creative rails: its story beats already carry the
    # stage, the performance action and the rig light for every segment.
    creative = approach.startswith("Creative") or approach.startswith("Concert")
    recipe = _shot_recipe_smart(idx, music=music)
    shot_a = recipe["shot"]
    shot_b = next_shot or _shot_recipe_smart(idx + 1, music=music)["shot"]

    lyric = seg.get("lyrics", "")
    imagery, mood = _lyric_imagery(lyric)
    mood = mood or theme["mood"]

    is_no_people = (who == "no_people" or subject_mode == "No People (Scenery & Objects only)")
    subject = "the scenery" if is_no_people else ((who or "").strip() or "the central figure")
    where = (setting or "").strip() or "the scene"

    # The story beat moves the segment's location and action forward, so the
    # clips advance the same journey the cards tell.
    beat = seg.get("_story") or {}
    if beat.get("location_desc"):
        where = beat["location_desc"]

    cam_move = _music_driven_camera(music, idx)
    parts = [f"{cam_move}, transition from {_an(shot_a)} to {_an(shot_b)}."]

    motion_action = _music_driven_motion(music, idx, who, setting, subject_mode)
    if beat.get("action") and not is_no_people and not abstract:
        motion_action = beat["action"]

    if abstract:
        # Music first: the sound picks the colour and form, a lyric only fills
        # in when the music has no opinion. No subject, no location - naming
        # them would pull the render back toward a literal shot.
        a_mood = _TONE_TO_MOOD.get((music or {}).get("emotional_tone", ""), "") or mood
        colour, form, motion = ABSTRACT_MOODS.get(a_mood, ABSTRACT_MOODS[""])
        parts.append(f"An abstract, painterly interpretation - no people, no faces, "
                     f"no recognisable place: {colour}; {form}.")
        parts.append(motion.capitalize() + ".")
        if imagery:
            parts.append("Echoes of " + imagery[0] + " surface and dissolve.")
    elif creative:
        if is_no_people:
            parts.append(f"In {where}, {motion_action}.")
        else:
            parts.append(f"{subject} in {where}, {motion_action}.")

        if lyric == INSTRUMENTAL:
            parts.append("Instrumental passage: the scene itself carries the dynamic motion.")
        else:
            pool = list(imagery)
            leak = [im for im in theme["imagery"] if im not in pool]
            if leak:
                pool.append(leak[idx % len(leak)])
            if pool:
                parts.append("Moving in the shot: " + ", ".join(pool[:3]) + ".")
        parts.append(CREATIVE_CINEMA[(idx * 5 + 2) % len(CREATIVE_CINEMA)].capitalize() + ".")
        if mood:
            parts.append(f"The whole shot feels {mood}.")
    else:  # literal
        if is_no_people:
            parts.append(f"In {where}, {motion_action}.")
        else:
            parts.append(f"{subject} in {where}, {motion_action}.")

        if lyric == INSTRUMENTAL:
            parts.append("Instrumental passage: no singing, environment carries the motion.")
        elif imagery:
            parts.append("Moving in the shot: " + ", ".join(imagery) + ".")
        if mood:
            parts.append(mood.strip().rstrip(".") + ".")

    if not abstract:
        world = _music_world(music)
        if world:
            parts.append(f"The scene is set in {world}.")

    pacing = _beat_pacing(music)
    if pacing:
        parts.append(pacing)

    ambience = f"{recipe['atmos']} drifting across the frame, {recipe['light']}."
    if beat.get("light") and not abstract:
        ambience = f"Time and light: {beat['light']}. " + ambience
    parts.append(ambience)

    extra = _video_style(motion_style)
    if extra:
        parts.append(extra + ".")
    if video_style:
        parts.append(video_style + ".")

    if abstract:
        subject_continuity = "The palette, forms and colour grade remain consistent throughout."
    elif is_no_people:
        subject_continuity = "The environment, lighting and colour grade remain consistent throughout."
    else:
        subject_continuity = "The same character, wardrobe, lighting and colour grade throughout."

    parts.append(
        f"One continuous {seg.get('duration', MIN_SEGMENT_SEC):.1f} second take with real "
        f"physical movement from start to finish - no cuts, no crossfade, no dissolve, no "
        f"morphing or warping between frames. {subject_continuity}")
    return " ".join(p for p in parts if p)



# ---------------------------------------------------------------------------
# Lyric -> imagery.
#
# Pasting a sung line straight into an image prompt does not work: "prom-photed
# coffee, ready? Human on the wheel" is not a picture. These tables turn what a
# segment is ABOUT into concrete things a camera could photograph, so each card
# becomes a visual representation of that part of the song.

# Filler that carries no image at all.
LYRIC_STOPWORDS = frozenset("""
a an the and or but if then than so as at by for from in into of on onto to with without
i im i'm me my mine we us our you your yours he him his she her hers it its they them their
is am are was were be been being do does did done have has had will would can could shall
should may might must not no nor yes yeah yea oh ooh ah aah hey hmm uh um la na
this that these those there here what when where who whom which why how all any some every
just only very really too also again still even ever never always now then once
gonna wanna gotta aint ain't cause cuz got get gets getting go goes going went come comes
say says said tell tells told know knows knew think thinks thought like likes liked
bye byebye yo uh-huh mmm woah whoa
it's that's there's what's who's she's he's let's don't won't can't didn't doesn't isn't
wasn't weren't couldn't wouldn't shouldn't i've you've we've they've i'd you'd we'd they'd
i'll you'll he'll she'll we'll they'll it'll y'all
same thing things one want wants wanted need needs stuff
""".split())

# word roots -> something you can actually photograph
LYRIC_IMAGERY = (
    (("ai", "agent", "agents", "bot", "bots", "robot", "android", "machine", "neural",
      "model", "models", "algorithm", "automation"),
     "a translucent holographic AI figure standing just behind them"),
    (("code", "coding", "coder", "logic", "script", "syntax", "compile", "commit", "commits",
      "import", "imports", "library", "libraries", "boilerplate", "debug", "bug", "bugs",
      "flaw", "patch", "repo", "git", "function", "loop", "stack"),
     "streams of glowing green code raining down through the air"),
    (("data", "file", "files", "scroll", "session", "server", "cloud", "network", "system",
      "instruction", "instructions", "prompt", "prompts", "token", "tokens"),
     "floating translucent data panels casting light on their face"),
    (("test", "tests", "testing", "check", "guard", "rail", "rails", "control", "boundaries",
      "safe", "safety", "secure"),
     "a glowing containment grid of light lines around them"),
    (("wheel", "ride", "rides", "riding", "drive", "driving", "road", "steady", "car", "truck",
      "highway", "lane", "steer"),
     "hands gripping a steering wheel, headlights streaking past the windscreen"),
    (("run", "running", "race", "chase", "fast", "speed", "rush", "escape", "flee"),
     "motion-blurred sprint through the frame, trailing light"),
    (("fly", "flying", "flight", "wings", "soar", "sky", "cloud", "clouds", "air"),
     "figure suspended midair against a vast open sky"),
    (("fire", "fireball", "burn", "burning", "flame", "flames", "spark", "sparks", "ember",
      "explode", "explosion", "blast"),
     "showers of orange sparks and drifting embers"),
    (("water", "rain", "ocean", "sea", "seas", "river", "wave", "waves", "flood", "drown",
      "tide", "storm"),
     "heavy rain sheeting down, water pooling and rippling underfoot"),
    (("night", "dark", "darkness", "midnight", "shadow", "shadows", "black"),
     "deep night, pools of darkness broken by hard pools of light"),
    (("light", "lights", "neon", "glow", "glowing", "shine", "bright", "flash", "spotlight",
      "star", "stars", "sun", "dawn", "sunrise"),
     "hard neon light sources flaring directly into the lens"),
    (("coffee", "cup", "mug", "drink", "bottle", "glass", "vodka", "whiskey", "wine", "bar"),
     "a steaming cup held close, catching the light"),
    (("heart", "love", "kiss", "hold", "touch", "close", "warm", "together"),
     "an intimate close hold, faces almost touching"),
    (("cry", "tears", "pain", "hurt", "broken", "break", "lost", "lonely", "alone", "empty",
      "fall", "falling", "down"),
     "turned away, shoulders down, isolated in a large empty space"),
    (("fight", "war", "battle", "strike", "hit", "punch", "sword", "gun", "blood", "enemy"),
     "braced mid-confrontation, tension in every muscle"),
    (("dream", "dreams", "sleep", "wake", "mind", "memory", "vision", "imagine"),
     "the scene dissolving into a dreamlike double exposure"),
    (("dance", "dancing", "music", "song", "sing", "beat", "rhythm", "drum", "guitar", "band"),
     "mid-dance, body caught in a sweep of stage light"),
    (("city", "street", "streets", "town", "building", "window", "door", "room", "house",
      "wall", "roof", "bridge"),
     "towering city architecture crowding the frame"),
    (("money", "gold", "rich", "crown", "king", "queen", "win", "winner", "throne"),
     "gilded light and reflective gold surfaces"),
    (("time", "clock", "hour", "day", "days", "year", "forever", "wait", "late"),
     "long exposure light trails suggesting time slipping past"),
    (("cold", "ice", "snow", "winter", "frost", "freeze"),
     "frost in the air, breath fogging"),
    (("road", "journey", "travel", "far", "away", "leave", "leaving", "gone", "home"),
     "a road stretching away to a distant vanishing point"),
    (("human", "people", "crowd", "face", "eyes", "hand", "hands", "body", "soul"),
     "a tight human detail - eyes, hands - filling the frame"),
    (("work", "job", "build", "building", "make", "made", "create", "craft", "tool", "tools"),
     "hands working, tools and sparks in close detail"),
)

# how a segment should FEEL when no concrete imagery is found
LYRIC_MOOD = (
    (("love", "heart", "warm", "hold", "together", "home", "smile"), "tender and warm"),
    (("lost", "alone", "cry", "pain", "broken", "empty", "cold", "gone"), "desolate and aching"),
    (("fight", "war", "burn", "hit", "break", "rage", "fury"), "violent and urgent"),
    (("run", "fast", "fly", "rise", "up", "high", "free", "wild"), "euphoric and soaring"),
    (("dream", "sleep", "float", "slow", "soft", "quiet"), "hazy and weightless"),
)

_WORD_RE = re.compile(r"[a-z']+")


def _lyric_words(lyric):
    """Whisper writes things like 'prom -photed' and 'pre -hello'; normalise before matching."""
    text = str(lyric).lower().replace("-", " ").replace("'", "'")
    return [w.strip("'") for w in _WORD_RE.findall(text) if len(w.strip("'")) > 1]


def _analyse_lyric_deep(lyric):
    """
    Adaptive Natural Language Analysis for lyrics.
    Extracts action verbs, visual nouns, sensory descriptors, and emotional tone
    without relying strictly on rigid static word tables. Synthesizes concrete
    visual imagery for any lyric line.
    """
    if not lyric or lyric == INSTRUMENTAL:
        return [], ""

    text = str(lyric).lower().replace("-", " ")
    words = [w.strip("',.!?\"()") for w in _WORD_RE.findall(text)]
    content_words = [w for w in words if w not in LYRIC_STOPWORDS and len(w) > 2]

    if not content_words:
        return [], ""

    # Check known imagery hints first
    picked, seen = [], set()
    for roots, image in LYRIC_IMAGERY:
        if set(content_words) & set(roots) and image not in seen:
            picked.append(image)
            seen.add(image)
            if len(picked) >= 3:
                break

    # Adaptive NLP fallback: extract concrete nouns & active verbs dynamically
    if not picked:
        visual_nouns = [w for w in content_words if w.endswith(("s", "y", "e", "r", "n", "t")) or len(w) > 4]
        action_verbs = [w for w in content_words if w.endswith(("ing", "ed", "es"))]

        if visual_nouns:
            main_nouns = visual_nouns[:2]
            if action_verbs:
                picked.append(f"{' '.join(main_nouns)} {action_verbs[0]} in the frame")
            else:
                picked.append(f"visual detail of {' and '.join(main_nouns)}")
        elif action_verbs:
            picked.append(f"dynamic movement of {action_verbs[0]}")

    mood = ""
    for roots, feel in LYRIC_MOOD:
        if set(content_words) & set(roots):
            mood = feel
            break

    if not mood and content_words:
        # Dynamic mood synthesis from word length & percussive sound
        if any(w.startswith(("dark", "night", "cold", "pain", "rain", "shadow", "ghost")) for w in content_words):
            mood = "desolate and aching"
        elif any(w.startswith(("sun", "light", "love", "gold", "fly", "fire", "star")) for w in content_words):
            mood = "euphoric and soaring"
        else:
            mood = "cinematic and evocative"

    return picked, mood


def _lyric_imagery(lyric, limit=3):
    """Concrete visual elements suggested by this segment's words (using adaptive NLP)."""
    return _analyse_lyric_deep(lyric)



def _lyric_keywords(lyric, limit=5):
    """The few words that actually carry meaning - used as a light hint only."""
    out, seen = [], set()
    for w in _lyric_words(lyric):
        if w in LYRIC_STOPWORDS or w in seen or len(w) < 3:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return out


# Models with a large language-model text encoder (Z-Image, FLUX, Qwen) follow a
# described scene far better than comma-separated tags; SDXL is the opposite.
PROSE_MODELS = {"Z-Image", "FLUX.1-Dev", "Qwen-Image", "Custom"}


# The measured emotional tone of the track, mapped onto the abstract palette
# pools - so an instrumental song still picks colour, form and motion energy.
_TONE_TO_MOOD = {
    "triumphant_euphoric": "euphoric and soaring",
    "intense_aggressive": "violent and urgent",
    "peaceful_intimate": "tender and warm",
    "melancholic_aching": "desolate and aching",
    "driving_kinetic": "violent and urgent",
    "balanced_cinematic": "",
}


def _abstract_card_prompt(index, seg, mood, music, visual_style, global_prompt, is_finale):
    """The Abstract approach: NO subject, NO location, NO literal scene. Colour,
    form and motion energy stand in for the music; a lyric may only surface as
    an echo. This is a different kind of image entirely, not the literal card
    with an 'abstract' sticker on it."""
    colour, form, motion = ABSTRACT_MOODS.get(mood, ABSTRACT_MOODS[""])
    palette = (music or {}).get("palette", "")
    lines = []
    lead = (global_prompt or "").strip().rstrip(".")
    if lead:
        lines.append(lead[0].upper() + lead[1:] + ".")
    lines.append(
        "An abstract non-representational composition - no people, no faces, "
        "no recognisable place, no objects, no text.")
    lines.append(f"{colour[0].upper() + colour[1:]}; {form}.")
    if palette:
        lines.append(f"Colour treatment: {palette}.")
    section_type = seg.get("section_type")
    if section_type == "chorus_climax":
        lines.append("The forms surge to peak intensity - maximum contrast, the frame full.")
    elif section_type == "build_up":
        lines.append("The forms gather and rise, tension building toward the edge of the frame.")
    elif section_type in ("intro", "outro"):
        lines.append("The forms are sparse and quiet, most of the frame breathing empty.")
    if is_finale:
        lines.append("The final resolving image: the forms settle into stillness.")
    imagery, _ = ([], "") if seg.get("lyrics") == INSTRUMENTAL else _lyric_imagery(seg.get("lyrics", ""))
    if imagery:
        lines.append(f"A faint echo of {imagery[index % len(imagery)]} half-forms in the shapes and dissolves.")
    lines.append(motion[0].upper() + motion[1:] + ", frozen at its most charged instant.")
    if (visual_style or "").strip():
        style = visual_style.strip().rstrip(".")
        lines.append(style[0].upper() + style[1:] + ".")
    lines.append("Gestural, painterly, bold - highly detailed texture, no text, no watermark.")
    return " ".join(lines)


def _build_card_prompt(index, card_count, seg, visual_style, model_tokens, is_finale, vary,
                       subject="", setting="", model="Z-Image", global_prompt="",
                       music=None, subject_mode=SUBJECT_MODES[0],
                       approach=PROMPT_APPROACHES[0], theme=None):
    """
    Build a picture of what this part of the song is about - driven by lyrics, music analysis, and subject mode.
    """
    lyric = seg["lyrics"]
    theme = theme or {"mood": "", "imagery": [], "keywords": []}
    abstract = approach.startswith("Abstract")
    # Concert rides the creative rails: its story beats already carry the
    # stage, the performance action and the rig light for every segment.
    creative = approach.startswith("Creative") or approach.startswith("Concert")
    is_no_people = (subject == "no_people" or subject_mode == "No People (Scenery & Objects only)")
    subject_desc = "scenery and environment" if is_no_people else ((subject or "").strip() or "a lead figure")
    setting = (setting or "").strip()
    recipe = _shot_recipe_smart(index, music=music) if vary else _shot_recipe_smart(0, music=music)

    instrumental = (lyric == INSTRUMENTAL)
    imagery, mood = ([], "") if instrumental else _lyric_imagery(lyric)
    keywords = [] if instrumental else _lyric_keywords(lyric)
    mood = mood or theme.get("mood") or _TONE_TO_MOOD.get(
        (music or {}).get("emotional_tone", ""), "")
    world = _music_world(music)

    # Abstract is its own kind of image - it shares nothing with the literal
    # card except the style lead, so it branches out before the shared build.
    # The MUSIC picks its colour and form; the lyric only adjusts if the music
    # has no opinion - atmosphere from the sound, content from the words.
    if abstract:
        tone_mood = _TONE_TO_MOOD.get((music or {}).get("emotional_tone", ""), "")
        prompt = _abstract_card_prompt(index, seg, tone_mood or mood, music,
                                       visual_style, global_prompt, is_finale)
        seg["_visual_note"] = (f"{'final frame' if is_finale else f'shot {index + 1}'}: "
                               f"abstract [{mood or 'neutral'}]")
        return prompt

    # The story beat carries what this card actually SHOWS: the action at this
    # point of the arc, the place the journey has reached, and where the light
    # is on the song's clock. It beats every generic fallback below.
    beat = seg.get("_story") or {}
    if beat and not is_no_people:
        beat_action = ("at rest at last, the story resolved in the final image"
                       if is_finale else beat["action"])
    elif creative and not instrumental and not is_no_people:
        beat_action = CREATIVE_ACTIONS[(index * 3 + 1) % len(CREATIVE_ACTIONS)]
    elif is_no_people:
        beat_action = "dynamic environment with natural atmospheric motion"
    elif instrumental:
        beat_action = "holding still in the scene, letting the musical phrase breathe"
    elif mood:
        beat_action = f"caught in a {mood} moment"
    else:
        beat_action = "mid-motion, held in the musical moment"

    if beat.get("location_desc"):
        setting = beat["location_desc"]
    if beat.get("motif") and beat["motif"] not in imagery:
        imagery = list(imagery) + [beat["motif"]]

    # The creative approach lets imagery from OTHER verses leak into this shot.
    if creative and not instrumental:
        leak = [im for im in theme.get("imagery", []) if im not in imagery]
        if leak:
            imagery = list(imagery) + [leak[index % len(leak)]]

    role = "the final frame of the music video" if is_finale else f"shot {index + 1} of {card_count}"

    if model in PROSE_MODELS:
        lines = []
        style_lead = (global_prompt or "").strip().rstrip(".")
        if style_lead:
            lines.append(style_lead[0].upper() + style_lead[1:] + ".")
        article = "An" if recipe["shot"][0].lower() in "aeiou" else "A"
        if is_no_people:
            lines.append(f"{article} {recipe['shot']} of {setting or 'the scene'}, no people, {beat_action}.")
        else:
            lines.append(f"{article} {recipe['shot']} of {subject_desc}, {beat_action}")
            if setting:
                lines[-1] += f", in {setting}"
            lines[-1] += "."
        if world:
            lines.append(f"The video lives in {world}.")
        if beat.get("light"):
            lines.append(f"Time and light: {beat['light']}.")
        section_type = seg.get("section_type")
        if section_type == "chorus_climax":
            lines.append("Peak intensity climax moment: vibrant rim lighting, maximum contrast, high visual energy.")
        elif section_type == "build_up":
            lines.append("Escalating build-up moment: rising volumetric light, accelerating atmospheric movement.")
        elif section_type == "intro":
            lines.append("Introductory atmospheric framing: gentle diffused light, wide establishing perspective.")
        elif section_type == "outro":
            lines.append("Resolving outro moment: soft fading light, peaceful continuous drift.")

        if imagery:
            lines.append("In the frame: " + "; ".join(imagery) + ".")
        if creative:
            lines.append(CREATIVE_CINEMA[(index * 5 + 2) % len(CREATIVE_CINEMA)].capitalize()
                         + ", caught mid-change in this still.")
        lines.append(f"{recipe['comp'].capitalize()}, {recipe['light']}, {recipe['atmos']}.")
        if visual_style.strip():
            style = visual_style.strip().rstrip(".")
            lines.append(style[0].upper() + style[1:] + ".")
        lines.append("Highly detailed, strong composition, sharp focus, no text, no watermark.")
        prompt = " ".join(lines)
    else:
        parts = []
        if (global_prompt or "").strip():
            parts.append(global_prompt.strip())
        parts += [recipe["shot"]]
        if not is_no_people:
            parts.append(subject_desc)
        parts.append(beat_action)
        parts += imagery
        if setting:
            parts.append(setting)
        if world:
            parts.append(world)
        if beat.get("light"):
            parts.append(beat["light"])
        if creative:
            parts.append(CREATIVE_CINEMA[(index * 5 + 2) % len(CREATIVE_CINEMA)])
        parts += [recipe["comp"], recipe["light"], recipe["atmos"]]
        if visual_style.strip():
            parts.append(visual_style.strip())
        parts.append("highly detailed, strong composition, sharp focus")
        if model_tokens:
            parts.append(model_tokens)
        prompt = ", ".join(p for p in parts if p)

    seg["_visual_note"] = (f"{role}: " + (", ".join(keywords) if keywords else "instrumental"))
    return prompt



def _card_sheet(segments, card_count, prompts, notes):
    """Card-by-card sheet: the lyric, what it was read as visually, and the prompt.
    This is how you check that a card really represents its part of the song."""
    bar = "=" * 78
    rows = [bar, " STORYBOARD CARDS - lyric -> visual reading -> prompt", bar,
            f" {card_count} cards for {len(segments)} segments", bar, ""]
    for i in range(card_count):
        finale = i >= len(segments)
        seg = segments[-1] if finale else segments[i]
        when = (f"{_fmt_time(seg['end_time'])} (final frame)" if finale
                else f"{_fmt_time(seg['start_time'])} - {_fmt_time(seg['end_time'])}")
        sec_str = f"   [{seg.get('section_label', '').upper()}]" if seg.get("section_label") else ""
        rows.append(f"[CARD {i:03d}]  {when}{sec_str}")
        lyric = seg.get("lyrics", "")
        if lyric == INSTRUMENTAL:
            rows.append("  LYRIC : (instrumental)")
        else:
            for j, line in enumerate(_wrap(lyric, 66, 0).split("\n")):
                rows.append(("  LYRIC : " if j == 0 else "          ") + line)
            imagery, mood = _lyric_imagery(lyric)
            reads = ", ".join(_lyric_keywords(lyric)) or "(no concrete words)"
            rows.append(f"  READS : {reads}" + (f"   [mood: {mood}]" if mood else ""))
            for img in imagery:
                rows.append(f"  DRAWS : {img}")
        for j, line in enumerate(_wrap(prompts[i], 66, 0).split("\n")):
            rows.append(("  PROMPT: " if j == 0 else "          ") + line)
        rows.append("")
    if notes:
        rows += [bar] + [" " + n for n in notes]
    rows.append(bar)
    return "\n".join(rows)


# --------------------------------------------------------------------------
# Character reference. A figure drifts from card to card unless every prompt
# describes it with the SAME words, so the song is analysed once, a reference
# sheet is written, and that exact wording is repeated on every card.

def _derive_reference(subject, setting, segments, global_prompt=""):
    """Build a character sheet from the subject. The subject alone decides the
    wardrobe - it is either typed by the user or derived from the lyrics, and
    nothing else may bolt clothing onto it. (A wardrobe keyword table used to
    live here; 'morning sun' matched its desert roots and dressed a country
    singer in a dust-worn denim jacket. Content must come from the song.)"""
    sheet = (subject or "").strip() or "a solitary figure"

    lead = (global_prompt or "").strip().rstrip(".")
    prompt = ""
    if lead:
        prompt += lead[0].upper() + lead[1:] + ". "
    prompt += (
        f"Character reference sheet of {sheet}. Single figure, centred, facing the camera, "
        f"neutral expression, full head and upper body clearly visible, even flat lighting, "
        f"plain uncluttered background, no props, nothing cropped. "
        f"This is the ONE character who must appear identical in every later shot: "
        f"same face, same hair, same wardrobe, same colours. "
        f"Highly detailed, sharp focus, no text, no watermark."
    )
    return sheet, prompt


# --------------------------------------------------------------------------
# Scene references. A song usually returns to a handful of places; rendering
# each one once and reusing it keeps the video from inventing a new world on
# every card.

SCENE_HINTS = (
    ("open_country", ("hunting", "hunt", "deer", "stand", "campfire", "bonfire", "creek",
                      "meadow", "barn", "farm", "farmhouse", "dirt", "county", "country",
                      "fishing", "fish", "lake", "pond", "reel", "rod", "bait", "dock",
                      "porch", "whiskey", "beer", "boots", "saddle", "tractor"),
     "open country at golden hour - misty fields and a treeline, a still lake catching "
     "the light, campfire smoke rising, a worn pickup on a dirt road"),
    ("city_night", ("neon", "city", "street", "streets", "downtown", "traffic", "sign", "signs",
                    "alley", "sidewalk", "taxi"),
     "a rain-slicked neon city street at night, glowing signage, wet reflective asphalt, "
     "distant traffic lights"),
    ("interior_room", ("room", "house", "home", "door", "window", "bed", "chair", "table",
                       "kitchen", "wall", "floor", "lamp"),
     "a sparse dimly lit interior room, one practical lamp, worn furniture, "
     "light falling through a window"),
    ("workspace", ("code", "screen", "monitor", "desk", "computer", "terminal", "keyboard",
                   "server", "office"),
     "a dark workspace lit only by monitors, cables and hardware, glowing screens, "
     "papers and coffee cups"),
    ("open_road", ("road", "highway", "drive", "car", "truck", "journey", "travel", "wheel",
                   "lane", "bridge"),
     "an empty highway stretching to the horizon at dusk, worn asphalt, roadside poles, "
     "vast open sky"),
    ("nature_wild", ("forest", "tree", "trees", "mountain", "field", "grass", "river",
                     "valley", "wood", "woods", "hill"),
     "a wild natural landscape, tall trees and undergrowth, mist between trunks, "
     "shafts of daylight"),
    ("water", ("sea", "ocean", "wave", "waves", "beach", "shore", "boat", "harbour",
               "harbor", "tide", "seas"),
     "a cold open sea under heavy sky, breaking waves, spray in the air, distant horizon"),
    ("stage", ("stage", "crowd", "concert", "microphone", "band", "spotlight", "show",
               "dance", "club"),
     "a concert stage in darkness, hard spotlights cutting through haze, "
     "a crowd silhouetted beyond"),
    ("winter", ("snow", "snowfall", "ice", "winter", "frost", "freeze", "frozen"),
     "a frozen winter landscape, snow drifting, muted blue-grey light, bare branches"),
    ("desert_sun", ("desert", "sun", "sand", "heat", "dust", "burn"),
     "a sun-scorched desert plain, heat shimmer, cracked earth, hard overhead light"),
    ("sky_flight", ("sky", "cloud", "clouds", "fly", "flying", "wings", "stars", "moon",
                    "space"),
     "a vast open sky above a cloud layer, cold high-altitude light, endless space"),
)


# Objects that have to look the same every time they appear. A location
# reference cannot carry these - a guitar, a car, a locket seen twice from two
# angles reads as two different objects unless it has a sheet of its own.
PROP_HINTS = (
    ("guitar", ("guitar", "guitars", "strings", "chord", "chords", "strum", "acoustic"),
     "the same guitar, its body shape, wood grain, scratches and hardware clearly readable"),
    ("car", ("car", "cars", "engine", "wheel", "wheels", "headlight", "headlights", "windscreen",
             "windshield", "drive", "driving"),
     "the same car, its make, colour, dents and number plate clearly readable"),
    ("motorcycle", ("bike", "motorcycle", "motorbike", "throttle", "handlebars"),
     "the same motorcycle, its frame, tank, colour and wear clearly readable"),
    ("train", ("train", "trains", "railway", "rails", "platform", "carriage", "station"),
     "the same train and platform, livery and lettering clearly readable"),
    ("boat", ("boat", "boats", "ship", "sail", "sails", "deck", "hull", "mast"),
     "the same boat, its hull, rigging and paintwork clearly readable"),
    ("phone", ("phone", "call", "calls", "calling", "screen", "message", "text", "ring"),
     "the same phone, its case, cracks and screen glow clearly readable"),
    ("letter", ("letter", "letters", "note", "notes", "paper", "page", "pages", "book",
                "write", "wrote", "written", "diary", "ink"),
     "the same handwritten letter and its paper, folds, ink and handwriting clearly readable"),
    ("mirror", ("mirror", "mirrors", "reflection", "reflect", "glass"),
     "the same mirror, its frame, spotting and the room it reflects clearly readable"),
    ("clock", ("clock", "clocks", "watch", "hour", "hours", "minute", "minutes", "time",
               "ticking", "hands"),
     "the same clock, its face, numerals and hands clearly readable"),
    ("flower", ("flower", "flowers", "rose", "roses", "petal", "petals", "bloom", "garden"),
     "the same flowers, their species, colour and stage of wilting clearly readable"),
    ("jewel", ("ring", "rings", "necklace", "locket", "chain", "pendant", "gold", "silver",
               "diamond", "jewel"),
     "the same piece of jewellery, its metal, stone and engraving clearly readable"),
    ("candle", ("candle", "candles", "lantern", "lamp", "flame", "wick", "torch"),
     "the same lamp or candle, its holder, wax and flame clearly readable"),
    ("mask", ("mask", "masks", "masked", "disguise", "face"),
     "the same mask, its material, shape and markings clearly readable"),
    ("key", ("key", "keys", "lock", "locked", "unlock", "door", "gate"),
     "the same key and lock, their metal, teeth and wear clearly readable"),
    ("bird", ("bird", "birds", "crow", "crows", "raven", "wing", "wings", "feather", "feathers"),
     "the same bird, its species, plumage and markings clearly readable"),
    ("blade", ("knife", "blade", "sword", "steel", "edge", "cut", "cuts"),
     "the same blade, its shape, handle and nicks clearly readable"),
)


# Who the song is about, read from what it keeps singing about. Only the ROLE
# comes from this table - wardrobe, colours and demeanour are composed from the
# music analysis at derive time, so the same role never dresses the same way in
# two different-sounding songs. (This table used to hold complete personas -
# "a young woman in a long coat" appeared in every city song regardless of how
# the song sounded.) Used only when 'subject' is left empty; typed always wins.
PERSONA_ROLES = (
    (("hunting", "hunt", "deer", "stand", "fishing", "fish", "reel", "rod", "bait", "truck",
      "beer", "whiskey", "boots", "dog", "campfire", "fire", "country", "barn", "farm",
      "porch", "creek", "lake", "boat"),
     "a countryman with a warm weathered face"),
    (("code", "coding", "screen", "server", "data", "ai", "agent", "agents", "compile",
      "keyboard", "terminal"),
     "a coder with focused tired eyes lit by screens"),
    (("stage", "crowd", "microphone", "band", "guitar", "sing", "singing", "song", "show"),
     "a singer carrying their instrument"),
    (("road", "highway", "drive", "driving", "wheel", "car", "miles", "lane"),
     "a lone driver, eyes on the horizon"),
    (("sea", "ocean", "sail", "sails", "wave", "waves", "shore", "harbour", "harbor", "tide"),
     "a weathered sailor with rope-worn hands"),
    (("dance", "dancing", "club", "beat", "floor", "dj"),
     "a dancer, movement written into every line of their body"),
    (("city", "streets", "neon", "rain", "midnight"),
     "a solitary figure moving through the city, thoughtful and unhurried"),
)


def _compose_persona(role, music):
    """Role from the lyrics + wardrobe from the sound = a subject built for THIS
    song, not pulled ready-made from a table."""
    wardrobe = LOOK_WARDROBE.get((music or {}).get("look", ""), "")
    return f"{role}, {wardrobe}" if wardrobe else role


def _derive_subject_setting(segments, music=None, subject_mode=SUBJECT_MODES[0]):
    """
    Read the song's lyrics, audio features, emotional tone, and subject_mode to dynamically
    synthesize a unique WHO (subject) and WHERE (setting) without static hardcoded templates.
    """
    words = {}
    for seg in segments:
        lyric = seg.get("lyrics", "")
        if lyric == INSTRUMENTAL:
            continue
        for w in set(_lyric_words(lyric)) - LYRIC_STOPWORDS:
            words[w] = words.get(w, 0) + 1

    lyric_text = " ".join(seg.get("lyrics", "") for seg in segments if seg.get("lyrics") != INSTRUMENTAL).lower()
    has_words = bool(words)

    # Audio signal descriptors for dynamic generation
    look = (music or {}).get("look", "cinematic_neutral")
    emo = (music or {}).get("emotional_tone", "balanced_cinematic").replace("_", " ")
    mode = (music or {}).get("mode", "major")
    tempo = float((music or {}).get("tempo", 120.0))
    brightness = float((music or {}).get("brightness", 0.25))

    # --- DYNAMIC SETTING SYNTHESIS ---
    setting = ""
    best_setting_score = 0
    for _sid, roots, desc in SCENE_HINTS:
        score = sum(words.get(r, 0) for r in roots)
        if score > best_setting_score:
            best_setting_score, setting = score, desc

    if not setting:
        # Dynamically build setting from audio parameters if no lyric hints
        if mode == "minor" and brightness < 0.2:
            setting = f"a shadowed {emo} landscape under a deep twilight sky, moody atmospheric haze"
        elif mode == "major" and brightness > 0.3:
            setting = f"a radiant {emo} setting bathed in warm golden light, expansive open atmosphere"
        elif tempo > 120:
            setting = f"an energetic {emo} environment with dynamic lighting contrast and kinetic movement"
        else:
            setting = f"a cinematic {emo} space with rich atmospheric depth and textured surfaces"

    # --- MODE-BASED SUBJECT SYNTHESIS ---
    if subject_mode == "No People (Scenery & Objects only)":
        return "no_people", setting

    if subject_mode == "Singer / Performer":
        vocal_style = "a passionate lead vocalist" if mode == "major" else "an expressive solo performer"
        return _compose_persona(f"{vocal_style} with dynamic stage presence and focused expression", music), setting

    if subject_mode == "Band / Group (Multiple People)":
        return _compose_persona("a multi-member band - guitarists, bassist, drummer and vocalist in rhythm", music), setting

    if subject_mode == "Crowd / Atmospheric People":
        return _compose_persona(f"an expressive crowd immersed in the music, reacting to the {tempo:.0f} BPM beat", music), setting

    if subject_mode == "Single Person / Main Character":
        best_p, persona_role = 0, ""
        for roots, role in PERSONA_ROLES:
            score = sum(words.get(r, 0) for r in roots)
            if score > best_p:
                best_p, persona_role = score, role
        role = persona_role or f"a protagonist whose presence carries the {emo} story"
        return _compose_persona(role, music), setting

    # --- AUTO MODE SYNTHESIS ---
    band_cues = ("band", "group", "guitarist", "drummer", "bassist", "musicians")
    if any(c in lyric_text for c in band_cues):
        return "a multi-member band performing together in rhythm", setting

    singer_cues = ("sing", "singer", "singing", "microphone", "mic", "vocal", "stage")
    if any(c in lyric_text for c in singer_cues):
        return "a lead singer performing with emotional intensity", setting

    crowd_cues = ("crowd", "party", "club", "dance", "dancing", "audience")
    if any(c in lyric_text for c in crowd_cues):
        return "an atmospheric crowd of people moving to the music", setting

    if not has_words or (look in ("ambient_dreamy", "cinematic_neutral") and not words):
        return "no_people", setting

    best_p, persona_role = 0, ""
    for roots, role in PERSONA_ROLES:
        score = sum(words.get(r, 0) for r in roots)
        if score > best_p:
            best_p, persona_role = score, role

    role = persona_role or f"a central figure whose journey reflects the {emo} tone of the song"
    return _compose_persona(role, music), setting


def _plan_references(subject, setting, segments, global_prompt="", max_scenes=6, max_props=4,
                     subject_mode=SUBJECT_MODES[0], story=None):
    """
    Returns (references, assignment). When a story is given, its journey
    locations ARE the scene references and each card is assigned the location
    its beat takes place in - so the reference sheet matches the story instead
    of re-deriving a different set of places from the same keywords.
    """
    lead = (global_prompt or "").strip().rstrip(".")
    lead_sentence = (lead[0].upper() + lead[1:] + ". ") if lead else ""
    setting = (setting or "").strip()

    references = []
    is_no_people = (subject == "no_people" or subject_mode == "No People (Scenery & Objects only)")

    if not is_no_people:
        character, character_prompt = _derive_reference(subject, setting, segments, global_prompt)
        references.append({
            "id": "character",
            "kind": "character",
            "name": character,
            "prompt": character_prompt,
        })

    # --- which places does the song keep visiting? ---
    hits = {}
    per_segment = {}
    for seg in segments:
        text = seg.get("lyrics", "")
        if text == INSTRUMENTAL:
            continue
        words = set(_lyric_words(text)) - LYRIC_STOPWORDS
        for scene_id, roots, _desc in SCENE_HINTS:
            overlap = len(words & set(roots))
            if overlap:
                hits[scene_id] = hits.get(scene_id, 0) + overlap
                per_segment.setdefault(seg["segment_index"], []).append((overlap, scene_id))

    if story:
        # The story already decided where the video happens; render exactly
        # those places, in journey order.
        scene_rows = [(sid, name, desc) for sid, name, desc in story["locations"]][:max_scenes]
    else:
        ranked = [sid for sid, _ in sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))][:max_scenes]
        if setting and "main_setting" not in ranked:
            ranked.append("main_setting")
        if not ranked:
            ranked = ["main_setting"]
        scene_rows = []
        for scene_id in ranked:
            if scene_id == "main_setting":
                scene_rows.append((scene_id, "the main setting", setting))
            else:
                description = next(d for sid, _r, d in SCENE_HINTS if sid == scene_id)
                if setting:
                    description = f"{description}; consistent with {setting}"
                scene_rows.append((scene_id, scene_id.replace("_", " "), description))

    for scene_id, name, description in scene_rows:
        references.append({
            "id": scene_id,
            "kind": "scene",
            "name": name,
            "prompt": (f"{lead_sentence}Establishing location reference: {description}. "
                       f"No people in frame. Wide, clear, evenly lit view of the place itself, "
                       f"showing its architecture, colours and materials. This exact location "
                       f"must stay recognisable in every shot set here. "
                       f"Highly detailed, sharp focus, no text, no watermark."),
        })

    # --- objects the song keeps coming back to ---
    prop_segments = {}
    for seg in segments:
        text = seg.get("lyrics", "")
        if text == INSTRUMENTAL:
            continue
        words = set(_lyric_words(text)) - LYRIC_STOPWORDS
        for prop_id, roots, _desc in PROP_HINTS:
            if words & set(roots):
                prop_segments.setdefault(prop_id, set()).add(seg["segment_index"])

    recurring = 1 if len(segments) <= 3 else 2
    prop_ids = sorted((p for p, segs in prop_segments.items() if len(segs) >= recurring),
                      key=lambda p: (-len(prop_segments[p]), p))[:max_props]
    for prop_id in prop_ids:
        description = next(d for pid, _r, d in PROP_HINTS if pid == prop_id)
        references.append({
            "id": prop_id,
            "kind": "prop",
            "name": prop_id.replace("_", " "),
            "prompt": (f"{lead_sentence}Object reference: {description}. "
                       f"The object alone, filling the frame against a plain neutral "
                       f"background, evenly lit, no people. This exact object must stay "
                       f"recognisable wherever it appears. "
                       f"Highly detailed, sharp focus, no text, no watermark."),
        })

    # --- assign one primary reference per card ---
    scene_ids = [r["id"] for r in references if r["kind"] == "scene"]
    has_character = any(r["kind"] == "character" for r in references)
    assignment = {}
    card_count = len(segments) + 1
    for i in range(card_count):
        seg_index = min(i, len(segments) - 1) if segments else 0
        best = None
        if story and seg_index < len(story["beats"]):
            cand = story["beats"][seg_index]["location_id"]
            if cand in scene_ids:
                best = cand
        for _overlap, scene_id in sorted(per_segment.get(seg_index, []), reverse=True):
            if best is not None:
                break
            if scene_id in scene_ids:
                best = scene_id
                break
        if best is None:
            best = "main_setting" if "main_setting" in scene_ids else (
                scene_ids[i % len(scene_ids)] if scene_ids else "main_setting")

        shot = _shot_recipe(i)["shot"]
        here = [p for p in prop_ids if seg_index in prop_segments.get(p, ())]
        if "extreme close-up" in shot and here:
            assignment[str(i)] = here[0]
        elif has_character and any(k in shot for k in ("close-up", "over-the-shoulder", "profile")):
            assignment[str(i)] = "character"
        else:
            assignment[str(i)] = best

    for prop_id in prop_ids:
        if prop_id in assignment.values():
            continue
        candidates = [i for i in range(card_count)
                      if (min(i, len(segments) - 1) if segments else 0)
                      in prop_segments.get(prop_id, ())]
        if candidates:
            tightest = max(candidates, key=lambda i: _shot_tightness(_shot_recipe(i)["shot"]))
            assignment[str(tightest)] = prop_id

    used = set(assignment.values())
    if has_character:
        used.add("character")
    references = [r for r in references if r["id"] in used]
    return references, assignment



def _reference_sheet(references, assignment, card_count):
    bar = "=" * 78
    by_kind = {}
    for ref in references:
        by_kind.setdefault(ref["kind"], []).append(ref)
    tally = ", ".join(f"{len(v)} {k}{'s' if len(v) != 1 else ''}"
                      for k, v in sorted(by_kind.items()))

    rows = [bar, " REFERENCES - rendered first in Part 2, then reused by the cards", bar,
            f" This song needs {len(references)} reference(s): {tally}.",
            " The lyrics decide these: every place and object the song keeps returning to",
            " gets its own sheet, and anything no card ends up using is dropped.", ""]
    counts = {}
    for i in range(card_count):
        rid = assignment.get(str(i), "character")
        counts[rid] = counts.get(rid, 0) + 1

    for ref in references:
        used = counts.get(ref["id"], 0)
        rows.append(f" [{ref['id']}]  ({ref['kind']})  {ref['name']}   ->  {used} card(s)")
        for line in _wrap(ref["prompt"], 70, 0).split("\n"):
            rows.append("     " + line)
        rows.append("")
    rows.append(" Cards per reference: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    rows.append(bar)
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Reference sheets.
#
# One picture of a character is not a reference - it only pins down the angle it
# happens to show. A turnaround (front, three-quarter, profile, back, portrait)
# describes the whole figure, so every card that leans on it is working from the
# same design rather than re-inventing the parts the single view never showed.
# ---------------------------------------------------------------------------

REFERENCE_VIEWS = {
    "character": [
        ("front", "full-body front view, standing squarely facing the camera, arms relaxed "
                  "at the sides, whole figure in frame from head to feet"),
        ("three_quarter", "full-body three-quarter view, the body turned about 45 degrees, "
                          "head turned the same way, whole figure in frame"),
        ("profile", "full-body side profile, standing exactly side-on to the camera, "
                    "whole figure in frame"),
        ("back", "full-body rear view seen from directly behind, whole figure in frame"),
        ("portrait", "head and shoulders portrait facing the camera, sharp facial detail, "
                     "hair and collar clearly readable"),
        ("detail", "close detail study of the wardrobe and any props carried, "
                   "fabric, texture and colour clearly readable"),
    ],
    "prop": [
        ("front", "straight-on front view of the object, filling the frame"),
        ("three_quarter", "three-quarter view of the object, turned about 45 degrees"),
        ("side", "exact side view of the object"),
        ("back", "rear view of the object"),
        ("detail", "macro detail of the object's most distinctive marking, wear or texture"),
        ("in_hand", "the object held in a pair of hands, showing its true scale"),
    ],
    "scene": [
        ("wide", "wide establishing view taking in the whole location"),
        ("reverse", "reverse angle, looking back across the location from the opposite side"),
        ("high", "high angle looking down over the location, layout clearly readable"),
        ("eye_level", "eye-level view standing inside the location"),
        ("detail", "close detail of the most characteristic corner of the location"),
        ("empty", "the location empty of people, architecture and set dressing clear"),
    ],
}

REFERENCE_SHEET_STYLE = {
    "character": ("character reference sheet, plain neutral grey studio backdrop, even soft "
                  "studio lighting with no dramatic shadows, the same face, the same "
                  "wardrobe and the same colours in every view, consistent character design"),
    "prop": ("object reference sheet, plain neutral background, even product lighting, the "
             "same shape, materials, colour and wear in every view, no people in frame "
             "except where hands are called for"),
    "scene": ("location reference, the same architecture, set dressing, time of day, weather "
              "and colour palette in every view, no people in frame"),
}


def _reference_views(kind, count):
    views = REFERENCE_VIEWS.get(kind) or REFERENCE_VIEWS["character"]
    return views[:max(1, min(int(count), len(views)))]


def _reference_view_prompt(ref, view_key, view_text):
    """The reference's own description, re-shot from one specific angle."""
    kind = ref.get("kind", "character")
    style = REFERENCE_SHEET_STYLE.get(kind, REFERENCE_SHEET_STYLE["character"])
    return f"{ref['prompt']}. {view_text}. {style}"


def _json_source_rows(data, raw="", title="STORYBOARD SOURCE - the JSON this node is reading"):
    """
    Which project.json a render node is actually working from.

    Part 2 and Part 3 read a timeline that was written by some earlier queue,
    and the single most confusing failure in the pipeline is rendering an old
    run without noticing. Every render node prints this block so the file, its
    write time and a fingerprint of the timeline are visible next to the work
    it produced.
    """
    run_dir = data.get("run_dir") or ""
    project_file = os.path.join(run_dir, "project.json") if run_dir else ""
    rows = ["=" * 78, f" {title}", "=" * 78]

    if project_file and os.path.isfile(project_file):
        st = os.stat(project_file)
        rows.append(f" file      : {project_file}")
        rows.append(f" written   : "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}"
                    f"   ({st.st_size:,} bytes on disk)")
    elif project_file:
        rows.append(f" file      : {project_file}")
        rows.append(" written   : NOT ON DISK - this data came straight from Part 1 in "
                    "the same queue, not from a saved run.")
    else:
        rows.append(" file      : (none) - storyboard_data was passed directly from Part 1, "
                    "no project.json involved.")

    rows.append(f" run       : {data.get('run_id', '-')}"
                f"   project '{data.get('project_name', '-')}'")
    rows.append(f" song      : {data.get('audio_path', '-')}")
    rows.append(f" timeline  : {_fmt_time(float(data.get('total_duration', 0.0) or 0.0))}"
                f"   {len(data.get('segments', []))} segments"
                f"   {data.get('card_count', '?')} cards"
                f"   lyrics from {data.get('lyric_source', '-')}")
    if raw:
        rows.append(f" data id   : crc32 {zlib.crc32(raw.encode('utf-8')) & 0xffffffff:08x}"
                    f"   ({len(raw):,} chars)  - same id means the same timeline")
    return rows


class StoryboardReferencesRender:
    """
    Renders every reference in the plan (the character, plus each location the
    song keeps returning to) in one queue. These come out first; the cards are
    then built on top of them so the video keeps one cast and one world.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_rerun": FORCE_RERUN_INPUT,
                "model": ("MODEL", ),
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "reference_plan": ("STRING", {"forceInput": True}),
                "views_per_reference": ("INT", {
                    "default": 4, "min": 1, "max": 6,
                    "tooltip": "Angles rendered per reference (front, three-quarter, profile, "
                               "back, portrait, detail). The first view is the one attached "
                               "to the cards.",
                }),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "text, watermark, logo, blurry, low quality, deformed, "
                               "extra limbs, bad anatomy",
                }),
            },
            "optional": {
                # Only used to name the project.json in the report, so you can
                # see which timeline these references belong to.
                "storyboard_data": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("REFERENCE_IMAGES", "REFERENCE_SHEET", "REFERENCE_PLAN",
                    "SHEET_LAYOUT", "REFERENCE_COUNT", "REFERENCE_REPORT")
    FUNCTION = "render_refs"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = ("Renders every reference in the plan as a turnaround sheet. "
                   "REFERENCE_IMAGES is the primary view of each reference, in plan order, "
                   "for attaching to cards; REFERENCE_SHEET is every angle.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan") if kwargs.get("force_rerun") else "cached"

    def render_refs(self, force_rerun, model, clip, vae, reference_plan, width, height,
                    steps, cfg, sampler_name, scheduler, seed, negative_prompt,
                    views_per_reference=4, storyboard_data=""):
        from comfy_execution.graph_utils import GraphBuilder

        _stage_cleanup("StoryboardReferencesRender")
        plan = json.loads(reference_plan)
        refs = plan.get("references") or []
        if not refs:
            raise ValueError("The reference plan is empty - run Part 1 and approve it first.")

        graph = GraphBuilder()
        negative = graph.node("CLIPTextEncode", text=negative_prompt, clip=clip)
        latent = graph.node("EmptyLatentImage", width=width, height=height, batch_size=1)

        primary = None      # view 0 of each reference, in plan order - cards use this
        sheet = None        # every view of every reference
        layout = []
        total = 0

        for i, ref in enumerate(refs):
            views = _reference_views(ref.get("kind", "character"), views_per_reference)
            for v, (view_key, view_text) in enumerate(views):
                text = _reference_view_prompt(ref, view_key, view_text)
                positive = graph.node("CLIPTextEncode", id=f"rpos{i}_{v}", text=text, clip=clip)
                sampler = graph.node(
                    "KSampler", id=f"rks{i}_{v}",
                    # Same seed for every view of one reference: the sampler then
                    # varies only by the angle in the prompt, which keeps the
                    # design of the figure steadier across the sheet.
                    model=model, seed=seed + i * 1000, steps=steps, cfg=cfg,
                    sampler_name=sampler_name, scheduler=scheduler,
                    positive=positive.out(0), negative=negative.out(0),
                    latent_image=latent.out(0), denoise=1.0,
                )
                decoded = graph.node("VAEDecode", id=f"rvd{i}_{v}", samples=sampler.out(0), vae=vae)

                layout.append({"index": total, "ref_id": ref["id"], "kind": ref.get("kind", "character"),
                               "name": ref.get("name", ref["id"]), "view": view_key,
                               "is_primary": v == 0, "seed": seed + i * 1000, "prompt": text})
                total += 1

                sheet = decoded if sheet is None else graph.node(
                    "ImageBatch", id=f"rsheet{i}_{v}", image1=sheet.out(0), image2=decoded.out(0))
                if v == 0:
                    primary = decoded if primary is None else graph.node(
                        "ImageBatch", id=f"rprim{i}", image1=primary.out(0), image2=decoded.out(0))

        try:
            src_data = json.loads(storyboard_data) if storyboard_data else {}
        except (ValueError, TypeError):
            src_data = {}

        ref_rows = []
        for row in layout:
            ref_rows.append(
                "-" * 78 + "\n"
                f" IMAGE {row['index']:03d}   [{row['ref_id']}] {row['kind']} - {row['name']}\n"
                f"   view      : {row['view']}"
                + ("   <- PRIMARY, this is the one the cards build on" if row["is_primary"] else "")
                + f"\n   seed      : {row['seed']}\n"
                f"   PROMPT SENT TO THE SAMPLER:\n"
                + "\n".join("     " + ln for ln in _wrap(row["prompt"], 68, 0).split("\n")))

        report = "\n".join(
            _json_source_rows(src_data, raw=storyboard_data,
                              title="REFERENCE RENDER - the JSON these references came from")
            + [f" canvas    : {width}x{height}   {steps} steps   cfg {cfg}   "
               f"{sampler_name}/{scheduler}",
               f" plan      : {len(refs)} reference(s) x {views_per_reference} view(s) "
               f"= {total} images",
               f" ids       : " + ", ".join(r["id"] for r in refs),
               "=" * 78,
               f" EVERY REFERENCE IMAGE AND THE EXACT PROMPT USED ({total} images)",
               "=" * 78]
            + ref_rows
            + ["=" * 78])
        _safe_print(report)

        run_dir = src_data.get("run_dir") or ""
        if run_dir and os.path.isdir(run_dir):
            try:
                out = os.path.join(run_dir, "reference_prompts.txt")
                with open(out, "w", encoding="utf-8") as handle:
                    handle.write(report)
                _safe_print(f"[StoryboardReferencesRender] Reference prompt report -> {out}")
            except Exception as exc:
                _safe_print(f"[StoryboardReferencesRender] Could not write the reference "
                            f"prompt report ({exc}).")

        return {"ui": {"text": (report,)},
                "expand": graph.finalize(),
                "result": (primary.out(0), sheet.out(0), reference_plan,
                           json.dumps(layout, indent=2), len(refs), report)}


class StoryboardReferenceSaver:
    """
    Writes the rendered reference sheet into the run's own references/ folder,
    named after the reference and the angle: character_front.png,
    character_profile.png, scene_rooftop_wide.png ...

    Reference images used to exist only as tensors passing between two nodes, so
    the sheet the whole video's look was built on vanished the moment the queue
    finished. Kept on disk they can be re-used, hand-picked, or fed back in.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "sheet_layout": ("STRING", {"forceInput": True}),
                "run_dir": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("IMAGES", "REFERENCES_DIR", "SAVE_REPORT")
    FUNCTION = "save_references"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = "Saves every reference angle into <run>/references/ as a named sheet."

    def save_references(self, images, sheet_layout, run_dir):
        from PIL import Image

        folder = _run_references_dir(run_dir)
        try:
            layout = json.loads(sheet_layout)
        except Exception:
            layout = []

        saved = []
        for i in range(images.shape[0]):
            entry = layout[i] if i < len(layout) else {}
            ref_id = _safe_name(entry.get("ref_id", f"reference_{i}"), f"reference_{i}")
            view = _safe_name(entry.get("view", f"view_{i}"), f"view_{i}")
            name = f"{ref_id}__{view}.png"
            array = np.clip(images[i].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(array).save(os.path.join(folder, name), compress_level=4)
            saved.append((name, entry))

        # A plain-text sheet beside the images, so the folder explains itself.
        index_path = os.path.join(folder, "references.txt")
        with open(index_path, "w", encoding="utf-8") as fh:
            fh.write(f"REFERENCE SHEET - {len(saved)} image(s)\n")
            fh.write(f"{GAP_AUTHOR}\n\n")
            for name, entry in saved:
                fh.write(f"{name}\n")
                fh.write(f"    reference : {entry.get('name', '-')}  ({entry.get('kind', '-')})\n")
                fh.write(f"    view      : {entry.get('view', '-')}"
                         + ("   <- attached to the cards\n" if entry.get("is_primary") else "\n"))
                fh.write(f"    prompt    : {_wrap(entry.get('prompt', '-'), 66, 16)}\n\n")

        by_ref = {}
        for _, entry in saved:
            by_ref.setdefault(entry.get("name", "?"), []).append(entry.get("view", "?"))

        bar = "=" * 78
        rows = [bar, " REFERENCE SHEET SAVED", bar, f" Folder : {folder}",
                f" Images : {len(saved)}", "-" * 78]
        for who, views in by_ref.items():
            rows.append(f" {who}")
            rows.append(f"     {', '.join(views)}")
        rows += ["-" * 78, " references.txt lists every angle and the prompt behind it.", bar]
        report = "\n".join(rows)
        _safe_print(report)
        return {"ui": {"text": (report,)}, "result": (images, folder, report)}


class StoryboardCardsRender:
    """
    Renders EVERY storyboard card in one queue.

    The timeline says how many cards are needed (N segments -> N+1 cards) and this
    node expands itself into that many sampler chains at execution time, so ten
    cards means ten renders - no batch counts, no manual index stepping.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_rerun": FORCE_RERUN_INPUT,
                "model": ("MODEL", ),
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "prompt_list": ("STRING", {"forceInput": True}),
                "storyboard_data": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 720, "min": 256, "max": 4096, "step": 32}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "text, watermark, logo, blurry, low quality, jpeg artifacts, "
                               "deformed, extra limbs, bad anatomy, oversaturated",
                }),
                "max_cards": ("INT", {"default": 64, "min": 1, "max": 512,
                                      "tooltip": "Safety cap. A long song needs many cards; raise if the deck is truncated."}),
            },
            "optional": {
                "reference_images": ("IMAGE", ),
                "reference_plan": ("STRING", {"forceInput": True}),
                "reference_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How hard each card holds onto its reference image. "
                               "0 = ignore it, 1 = hug it closely (less freedom for the shot).",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("CARD_IMAGES", "CARD_COUNT", "CARD_REPORT")
    FUNCTION = "render_all"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = ("Renders all N+1 storyboard cards in a single queue by expanding into one "
                   "sampler chain per card. CARD_REPORT names the project.json being read and "
                   "prints the exact prompt, reference and seed used for every card.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("force_rerun"):
            return float("nan")
        return "cached"

    def render_all(self, force_rerun, model, clip, vae, prompt_list, storyboard_data,
                   width, height, steps, cfg, sampler_name, scheduler, seed,
                   negative_prompt, max_cards, reference_images=None, reference_plan=None,
                   reference_strength=0.5):
        from comfy_execution.graph_utils import GraphBuilder

        _stage_cleanup("StoryboardCardsRender")
        prompts = json.loads(prompt_list)
        data = json.loads(storyboard_data)
        required = data.get("card_count", len(data.get("segments", [])) + 1)
        count = min(len(prompts), required, max_cards)
        if count <= 0:
            raise ValueError("No card prompts to render - run the storyboard prompt generator first.")
        if required > max_cards:
            _safe_print(f"[StoryboardCardsRender] WARNING: timeline needs {required} cards but "
                        f"max_cards={max_cards}; only {count} will be rendered.")

        _safe_print(f"[StoryboardCardsRender] Expanding into {count} card renders "
                    f"({width}x{height}, {steps} steps) - this runs them all in one queue.")

        # Which reference does each card build on?
        ref_order, assignment, ref_by_id = [], {}, {}
        if reference_images is not None and reference_plan:
            plan = json.loads(reference_plan)
            ref_order = [r["id"] for r in (plan.get("references") or [])]
            ref_by_id = {r["id"]: r for r in (plan.get("references") or [])}
            assignment = plan.get("assignment") or {}
            available = int(reference_images.shape[0])
            if available < len(ref_order):
                _safe_print(f"[StoryboardCardsRender] WARNING: plan lists {len(ref_order)} "
                            f"references but only {available} image(s) were supplied; "
                            "the extras will fall back to the first reference.")
            # denoise: how much freedom the card has away from its reference
            denoise = max(0.35, min(1.0, 1.0 - 0.65 * float(reference_strength)))
            _safe_print(f"[StoryboardCardsRender] Building on references "
                        f"(strength {reference_strength:.2f} -> denoise {denoise:.2f}).")
        else:
            denoise = 1.0

        graph = GraphBuilder()
        negative = graph.node("CLIPTextEncode", text=negative_prompt, clip=clip)
        latent = graph.node("EmptyLatentImage", width=width, height=height, batch_size=1)

        segments = data.get("segments", [])
        card_rows = []

        combined = None
        for i in range(count):
            positive = graph.node("CLIPTextEncode", id=f"pos{i}", text=prompts[i], clip=clip)

            start = latent
            card_denoise = 1.0
            ref_label = "none - rendered from an empty latent"
            if ref_order:
                rid = assignment.get(str(i), ref_order[0])
                slot = ref_order.index(rid) if rid in ref_order else 0
                slot = min(slot, int(reference_images.shape[0]) - 1)
                ref = ref_by_id.get(rid) or {}
                ref_label = (f"[{rid}] {ref.get('kind', '?')} - {ref.get('name', rid)} "
                             f"(reference image #{slot})")
                picked = graph.node("ImageFromBatch", id=f"rp{i}",
                                    image=reference_images, batch_index=slot, length=1)
                scaled = graph.node("ImageScale", id=f"rs{i}", image=picked.out(0),
                                    upscale_method="lanczos", width=width, height=height,
                                    crop="center")
                start = graph.node("VAEEncode", id=f"re{i}", pixels=scaled.out(0), vae=vae)
                card_denoise = denoise

            sampler = graph.node(
                "KSampler", id=f"ks{i}",
                model=model, seed=seed + i, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive.out(0), negative=negative.out(0),
                latent_image=start.out(0), denoise=card_denoise,
            )
            decoded = graph.node("VAEDecode", id=f"vd{i}", samples=sampler.out(0), vae=vae)
            if combined is None:
                combined = decoded
            else:
                combined = graph.node("ImageBatch", id=f"cat{i}",
                                      image1=combined.out(0), image2=decoded.out(0))

            # Everything that decided THIS card, recorded next to the prompt
            # that was actually encoded - so a wrong-looking card can be traced
            # without re-reading the JSON by hand.
            is_finale = i >= len(segments)
            seg = segments[i] if not is_finale else (segments[-1] if segments else {})
            when = (f"{_fmt_time(seg.get('start_time', 0))} -> {_fmt_time(seg.get('end_time', 0))}"
                    if seg and not is_finale else "final frame")
            card_rows.append(
                "-" * 78 + "\n"
                f" CARD {i:03d}   {when}   [{seg.get('section_label', '-') if seg else '-'}]\n"
                f"   seed      : {seed + i}    denoise {card_denoise:.2f}\n"
                f"   reference : {ref_label}\n"
                f"   lyric     : {_wrap(str(seg.get('lyrics', '-')) if seg else '-', 62, 15)}\n"
                f"   PROMPT SENT TO THE SAMPLER:\n"
                + "\n".join("     " + ln for ln in _wrap(prompts[i], 68, 0).split("\n")))

        report = "\n".join(
            _json_source_rows(data, raw=storyboard_data,
                              title="CARD RENDER - the JSON this deck was built from")
            + [f" prompts   : {len(prompts)} in prompt_list, rendering {count}"
               + (f" (capped by max_cards={max_cards})" if required > max_cards else ""),
               f" canvas    : {width}x{height}   {steps} steps   cfg {cfg}   "
               f"{sampler_name}/{scheduler}",
               f" seeds     : {seed} .. {seed + count - 1}   (one per card)",
               f" refs      : " + (f"{len(ref_order)} reference image(s), strength "
                                   f"{reference_strength:.2f} -> denoise {denoise:.2f}"
                                   if ref_order else "not connected - cards render from scratch"),
               "=" * 78,
               f" EVERY CARD AND THE EXACT PROMPT USED ({count} cards)",
               "=" * 78]
            + card_rows
            + ["=" * 78])
        _safe_print(report)

        # A copy next to the run, so the deck can be checked after the queue is
        # gone and diffed against the next render.
        run_dir = data.get("run_dir") or ""
        if run_dir and os.path.isdir(run_dir):
            try:
                out = os.path.join(run_dir, "card_prompts.txt")
                with open(out, "w", encoding="utf-8") as handle:
                    handle.write(report)
                _safe_print(f"[StoryboardCardsRender] Card prompt report -> {out}")
            except Exception as exc:
                _safe_print(f"[StoryboardCardsRender] Could not write the card prompt "
                            f"report ({exc}).")

        return {
            "ui": {"text": (report,)},
            "expand": graph.finalize(),
            "result": (combined.out(0), count, report),
        }


class StoryboardCardSelector:
    """
    Selects a specific storyboard card for previewing, tweaking prompt, or rerendering in pause workflow.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard_data": ("STRING", {"forceInput": True}),
                "prompt_list": ("STRING", {"forceInput": True}),
                "card_index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
            },
            "optional": {
                "override_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("CARD_INDEX", "CARD_PROMPT", "SEGMENT_LYRICS", "CARD_INFO")
    FUNCTION = "select_card"
    CATEGORY = "Geekatplay/VideoClipMaker"

    def select_card(self, storyboard_data, prompt_list, card_index, override_prompt=""):
        data = json.loads(storyboard_data)
        prompts = json.loads(prompt_list)

        idx = min(card_index, len(prompts) - 1)
        selected_prompt = override_prompt.strip() if override_prompt.strip() else prompts[idx]

        segments = data.get("segments", [])
        if idx < len(segments):
            lyrics = segments[idx]["lyrics"]
            info = f"Card {idx} (Start of Segment {idx} at {segments[idx]['start_time']}s)"
        else:
            lyrics = segments[-1]["lyrics"] if segments else ""
            info = f"Card {idx} (Final End Frame at {data.get('total_duration', 0)}s)"

        return (idx, selected_prompt, lyrics, info)


class KeyframePairBatcher(ForceRerunMixin):
    """
    Pairs consecutive rendered card images (Card i and Card i+1) dynamically for LTX 2.5 Image-to-Video generation.
    Unloads image generation models before video diffusion begins.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_rerun": FORCE_RERUN_INPUT,
                "images": ("IMAGE", ),
                "animation_prompts": ("STRING", {"forceInput": True}),
                "storyboard_data": ("STRING", {"forceInput": True}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1,
                                          "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "FLOAT", "INT", "BOOLEAN", "STRING", "INT")
    RETURN_NAMES = ("START_IMAGE", "END_IMAGE", "MOTION_PROMPT", "DURATION", "DURATION_INT",
                    "IS_LAST_SEGMENT", "SEGMENT_INFO", "SEGMENT_INDEX")
    FUNCTION = "get_pair"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"

    def get_pair(self, images, animation_prompts, storyboard_data, segment_index, force_rerun=False):
        # Explicitly unload image generator models before loading video generator (LTX 2.5) models
        if mm is not None:
            mm.unload_all_models()
            mm.soft_empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        data = json.loads(storyboard_data)
        anim_prompts = json.loads(animation_prompts)
        segments = data.get("segments", [])

        batch_size = images.shape[0]
        idx = min(segment_index, len(segments) - 1)

        start_img_idx = min(idx, batch_size - 1)
        end_img_idx = min(idx + 1, batch_size - 1)

        start_image = images[start_img_idx:start_img_idx+1]
        end_image = images[end_img_idx:end_img_idx+1]

        motion_prompt = anim_prompts[idx] if idx < len(anim_prompts) else "Smooth transition"
        duration = float(segments[idx]["duration"]) if idx < len(segments) else MIN_SEGMENT_SEC
        is_last = (idx >= len(segments) - 1)

        # LTX 2.5 takes whole seconds; feed DURATION_INT into its duration input so the
        # rendered clip is as long as the audio segment it covers.
        duration_int = int(round(max(MIN_SEGMENT_SEC, min(MAX_SEGMENT_SEC, duration))))

        # What is actually being handed to LTX 2.5 right now.
        total = len(segments)
        seg = segments[idx] if idx < total else {}
        bar = "=" * 78
        rows = [
            bar,
            f" FEEDING LTX 2.5   segment {idx + 1} of {total}",
            bar,
            f" Timecode  : {_fmt_time(seg.get('start_time', 0))} -> {_fmt_time(seg.get('end_time', 0))}",
            f" Length    : {duration:.2f}s  ->  LTX duration {duration_int}s",
            f" Frames    : card {start_img_idx}  ->  card {end_img_idx}   "
            f"(deck has {batch_size} cards)",
            f" Lyric     : {_wrap(seg.get('lyrics', '-'), 62, 13)}",
            "-" * 78,
            " MOTION PROMPT sent to LTX 2.5:",
        ]
        for line in _wrap(motion_prompt, 74, 0).split("\n"):
            rows.append("   " + line)
        rows.append("-" * 78)
        if segment_index != idx:
            # Clamped. Queueing again now just re-renders the last segment on top
            # of itself, which reads exactly like "it stopped after one segment".
            rows.append(f" *** segment_index is {segment_index}, but this project only has "
                        f"{total} segment(s). ***")
            rows.append(f" Re-rendering segment {idx} again. Set segment_index back to 0 "
                        f"and use a Queue")
            rows.append(f" batch count of {total} to render the song once, start to finish.")
        elif is_last:
            rows.append(f" This is the LAST segment ({total} of {total}).")
        else:
            remaining = total - (idx + 1)
            rows.append(f" {remaining} segment(s) still to render after this one.")
            rows.append(f" Set the Queue batch count to {total} and queue ONCE - 'segment_index'")
            rows.append(" steps itself. Queueing with a batch count of 1 renders only this segment,")
            rows.append(" and the final video would then be this clip alone.")
        rows.append(bar)
        info = "\n".join(rows)
        _safe_print(info)

        return {"ui": {"text": (info,)},
                "result": (start_image, end_image, motion_prompt, duration, duration_int,
                           is_last, info, idx)}


class GAPLastFrame:
    """The last frame of an image batch, as a 1-image batch.

    This is the continuity hinge of the whole video: LTX only *approximates*
    its end-frame guide, so if the next segment started from the original card
    the seam would jump from the approximation back to the card. Starting each
    segment from the previous segment's actual rendered last frame makes every
    seam pixel-continuous.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE", )}}

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("LAST_FRAME", )
    FUNCTION = "last_frame"
    CATEGORY = "Geekatplay/VideoClipMaker"

    def last_frame(self, images):
        return (images[-1:], )


class SegmentNowRendering:
    """
    Live per-segment banner inside LTXSegmentsRender's expansion.

    Sits in the image path at the head of each segment's chain and is chained
    behind the previous segment's saver, so it executes exactly when its segment
    starts. Being an OUTPUT_NODE, its text lands on the parent render node's
    panel - the canvas always shows WHICH segment is rendering right now, which
    cards it spans, and the motion prompt being sent to LTX 2.5.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_image": ("IMAGE", ),
                "end_image": ("IMAGE", ),
                "info": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "after": (ANY, ),
                # Sequencing only, like `after`: wiring the up-front frame
                # preview here makes every first/last frame load and display
                # BEFORE the first segment renders.
                "frames_ready": (ANY, ),
                # JSON telling the canvas which row is live right now.
                "board": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("START_IMAGE", "END_IMAGE")
    FUNCTION = "announce"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = "Announces the segment now being rendered; passes its two cards through."

    def announce(self, start_image, end_image, info, after=None, board="",
                 frames_ready=None):
        # Per-segment memory trace. A long render dies partway through because
        # memory creeps up segment by segment, and nothing else in the log shows
        # that creep - by the time it fails you cannot tell whether it drifted
        # or fell off a cliff. Printed for every segment so the last banner
        # before a crash carries the evidence.
        info = f"{info}\n MEMORY   : {_memory_line()}"
        _safe_print(info)
        warning = _memory_warning()
        if warning:
            _safe_print(warning)
            info += warning
        payload = {"text": (info,)}
        if board:
            payload["gap_board"] = (board,)
        return {"ui": payload, "result": (start_image, end_image)}


# What each video model can actually digest. The motion prompt builder stacks
# camera + action + imagery + world + pacing + light + continuity and lands
# around 210-225 words (~1350 chars); every model below does better with less.
#   LTX 2.5    - Gemma text encoder, guidance says a single detailed paragraph
#                of ~100-200 words; past that the style boilerplate dilutes the
#                motion instruction.
#   MiniMax H3 - concise conditioning; short direct prompts follow best.
VIDEO_PROMPT_BUDGETS = {
    "LTX 2.5": 1000,      # chars, ~160 words
    "MiniMax H3": 620,    # chars, ~100 words
}


def _fit_prompt(text, max_chars, keep_tail=1):
    """
    Fit a prose prompt into a model's budget without cutting mid-sentence.
    Sentences are kept from the front (camera, subject, action come first, so
    the most load-bearing content survives), later decorative sentences are
    dropped, and the last `keep_tail` sentence(s) are always kept - that is
    where the continuity clause lives ("one continuous take ... same character
    throughout"), which matters more than any flourish it displaces.
    """
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1 + keep_tail:
        return text[:max_chars].rsplit(" ", 1)[0]

    tail = sentences[-keep_tail:] if keep_tail else []
    tail_len = sum(len(s) + 1 for s in tail)
    kept, used = [], 0
    for sentence in sentences[:len(sentences) - keep_tail]:
        cost = len(sentence) + (1 if kept else 0)
        if used + cost + tail_len > max_chars:
            break
        kept.append(sentence)
        used += cost
    if not kept:                      # first sentence alone blows the budget
        kept = [sentences[0][:max(0, max_chars - tail_len)].rsplit(" ", 1)[0]]
    return " ".join(kept + tail)


def _segments_cache_key(kwargs):
    """
    Cache key for the segment renderers, which render the song a few segments
    per queue.

    Returning a constant would let ComfyUI serve the previous queue's cached
    result, so queueing again would render nothing and the song would stop
    after the first batch. Keying on the clips already written means every
    queue that has work left looks 'changed' and re-executes, while a finished
    song stays cached and costs nothing.
    """
    if kwargs.get("force_rerun") or not kwargs.get("resume", True):
        return float("nan")
    run_dir = kwargs.get("run_dir") or ""
    clips_dir = os.path.join(run_dir, "clips") if run_dir else ""
    try:
        have = sorted(f for f in os.listdir(clips_dir)
                      if f.startswith("clip_") and f.endswith(".mp4"))
    except OSError:
        have = []
    return f"{clips_dir}|{len(have)}|" + ",".join(have)


def _board_rows(segments, prompts, run_dir, fps=24, deck=None, queued=(),
                budget=None):
    """
    One row per segment for the canvas table: which two cards it animates, the
    prompt driving them, and its clip once that exists.

    Everything is read from the run folder, so the same rows can be produced by
    the render node while it works and by the table node on its own - the table
    then shows real progress without having to render anything.
    """
    cards_dir = os.path.join(run_dir, "cards") if run_dir else ""
    clips_dir = os.path.join(run_dir, "clips") if run_dir else ""
    cards = int(deck) if deck else (len(segments) + 1)
    queued = set(queued)
    rows = []
    for i, seg in enumerate(segments):
        idx = int(seg.get("segment_index", i))
        a = min(i, cards - 1)
        b = min(i + 1, cards - 1)
        duration = float(seg.get("duration", MIN_SEGMENT_SEC))
        clip_path = os.path.join(clips_dir, f"clip_{idx:03d}.mp4") if clips_dir else ""
        clip_done = bool(clip_path) and os.path.isfile(clip_path)
        # The URL is only for playback; DONE is decided by the file itself, so a
        # run outside ComfyUI's output folder still reports progress correctly.
        clip = _view_ref(clip_path)
        text = prompts[i] if i < len(prompts) else ""
        if text and budget:
            text = _fit_prompt(text, budget, keep_tail=2)
        rows.append({
            "i": i,
            "segment_index": idx,
            "time": (f"{_fmt_time(seg.get('start_time', 0))} -> "
                     f"{_fmt_time(seg.get('end_time', 0))}"),
            "dur": round(duration, 2),
            # LTX wants (frames - 1) divisible by 8; shown so the table matches
            # what the renderer will actually produce.
            "frames": max(9, int(round(duration * fps / 8.0)) * 8 + 1),
            "cardA": a,
            "cardB": b,
            "first": _view_ref(os.path.join(cards_dir, f"card_{a:04d}.png")) if cards_dir else None,
            "last": _view_ref(os.path.join(cards_dir, f"card_{b:04d}.png")) if cards_dir else None,
            "clip": clip,
            "prompt": text or "",
            "lyric": str(seg.get("lyrics", "")),
            "status": "done" if clip_done else ("queued" if i in queued else "pending"),
        })
    return rows


class SegmentTable(ForceRerunMixin):
    """
    Every segment of the song in one table on the canvas: the first and last
    card each clip is animated between, the motion prompt driving it, and the
    rendered clip once it exists.

    Queue it on its own at any time - before rendering to check the schedule,
    during to watch progress, after to review what came out. It reads the run
    folder, so it never re-renders anything.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_rerun": FORCE_RERUN_INPUT,
                "storyboard_data": ("STRING", {"forceInput": True}),
                "run_dir": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "animation_prompts": ("STRING", {"forceInput": True}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120,
                                "tooltip": "Only used to show the frame count each "
                                           "segment will render to."}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("TABLE", "RENDERED", "TOTAL")
    FUNCTION = "build_table"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = ("Shows every segment with its first/last card, its prompt and its "
                   "rendered clip. Queue it any time; it only reads the run folder.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Clips appear as the render progresses, so the table must re-read the
        # folder every queue instead of serving a cached picture of it.
        return float("nan")

    def build_table(self, storyboard_data, run_dir, force_rerun=False,
                    animation_prompts="", fps=24):
        data = json.loads(storyboard_data)
        segments = data.get("segments", [])
        deck = data.get("card_count", len(segments) + 1)
        try:
            prompts = json.loads(animation_prompts) if animation_prompts else []
        except Exception:
            prompts = []
        if not prompts:
            prompts = data.get("animation_prompts") or []

        rows = _board_rows(segments, prompts, run_dir, fps=fps, deck=deck,
                           budget=VIDEO_PROMPT_BUDGETS.get("LTX 2.5"))
        rendered = sum(1 for r in rows if r["status"] == "done")

        bar = "=" * 78
        lines = [bar, " SEGMENT TABLE", bar,
                 f" Run      : {os.path.basename(run_dir or '(none)')}",
                 f" Segments : {len(rows)}   cards {deck}   rendered {rendered}/{len(rows)}",
                 bar,
                 " SEG   TIME                  CARDS      CLIP"]
        for r in rows:
            lines.append(f" {r['i'] + 1:>3}   {r['time']:<20}  {r['cardA']:>3}->{r['cardB']:<3}  "
                         f"{'RENDERED' if r['status'] == 'done' else 'not rendered'}")
        missing = [r["i"] + 1 for r in rows if r["status"] != "done"]
        if missing:
            shown = ", ".join(str(m) for m in missing[:14]) + (", ..." if len(missing) > 14 else "")
            lines += ["-" * 78, f" Still to render: {shown}"]
        else:
            lines += ["-" * 78, " Every segment is rendered - the stitcher can build the final cut."]
        lines.append(bar)
        report = "\n".join(lines)
        _safe_print(f"[SegmentTable] {rendered}/{len(rows)} segment(s) rendered.")

        board = {"kind": "plan", "engine": "segment table",
                 "run": os.path.basename(run_dir or ""),
                 "total": len(rows), "batch": 0,
                 "done": rendered, "remaining": len(rows) - rendered, "rows": rows}
        return {"ui": {"text": (report,), "gap_board": (json.dumps(board),)},
                "result": (report, rendered, len(rows))}


class LTXSegmentsRender(ForceRerunMixin):
    """
    Renders the song's video segments, `segments_per_queue` at a time.

    The timeline says how many segments the song has, and this node expands
    itself into that many complete LTX 2.5 first/last-frame chains at execution
    time - the same mechanism StoryboardCardsRender uses to render every card.

    It renders a BATCH per queue rather than the whole song, because ComfyUI
    keeps every node output of a queue alive until that queue ends: a decoded
    six-second 1280x704 clip is over a gigabyte of frames, so a thirty-segment
    song expanded at once needs tens of gigabytes and dies with an out-of-memory
    error before the first clip is written. Each clip is saved as it finishes
    and skipped on the next queue, so queueing repeatedly walks the song from
    start to end with a bounded memory footprint.

    Each segment's chain mirrors the LTX 2.5 flf2v reference graph node for
    node: resize both cards -> LTXVPreprocess -> empty AV latents -> conditioning
    with the segment's motion prompt -> AddGuide(first, 0.7) -> AddGuide(last,
    end_guide_strength) -> DualCFG guider -> euler_ancestral over the distilled
    sigmas -> crop guides -> tiled video decode + audio decode -> CreateVideo.
    Every clip is saved into the run under its segment number as it finishes.
    """

    # The distilled LTX 2.5 schedule from the reference workflow (8 steps).
    DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_rerun": FORCE_RERUN_INPUT,
                "model": ("MODEL", {"tooltip": "LTX 2.5 transformer (UNETLoader)."}),
                "clip": ("CLIP", {"tooltip": "LTX 2.5 text encoder (CLIPLoader, type 'ltxv')."}),
                "video_vae": ("VAE", ),
                "audio_vae": ("VAE", ),
                "images": ("IMAGE", {"tooltip": "The full card deck from the card loader."}),
                "animation_prompts": ("STRING", {"forceInput": True}),
                "storyboard_data": ("STRING", {"forceInput": True}),
                "run_dir": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 704, "min": 256, "max": 4096, "step": 32}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": False,
                                 "tooltip": "Segment i renders with seed + i, so one seed drives the whole song reproducibly."}),
                "end_guide_strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How hard each clip is pulled onto the NEXT card. 0.7 matches "
                               "the official LTX 2.5 first/last-frame workflow, which pins "
                               "BOTH ends at 0.7, and is what makes segments join seamlessly: "
                               "clip i actually arrives at card i+1, which is exactly where "
                               "clip i+1 begins. Lowering it lets the clip drift away from "
                               "that card, so every segment boundary becomes a visible CUT. "
                               "Raise toward 1.0 only if you want a dissolve between cards.",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, out of focus, low contrast, washed out colors, "
                               "excessive noise, flickering, distorted proportions, deformed "
                               "facial features, extra limbs, jittery movement, unnatural "
                               "transitions, cartoonish rendering, AI artifacts",
                }),
                "max_segments": ("INT", {"default": 999, "min": 1, "max": 999,
                                         "tooltip": "Safety cap. Leave at 999 to render the whole song."}),
            },
            "optional": {
                "img_compression": ("INT", {"default": 18, "min": 0, "max": 50}),
                "segments_per_queue": ("INT", {
                    "default": 4, "min": 1, "max": 999,
                    "tooltip": "How many segments to render in ONE queue. ComfyUI keeps every "
                               "node's output alive until the queue ends, and a single decoded "
                               "clip is over a gigabyte, so expanding a whole song at once "
                               "exhausts memory and OOMs. Render a few at a time and queue "
                               "again - finished clips are skipped. Raise it only if you have "
                               "memory to spare.",
                }),
                "resume": ("BOOLEAN", {
                    "default": True,
                    "label_on": "skip clips already rendered",
                    "label_off": "re-render every segment",
                    "tooltip": "Skips any segment whose clip is already in the run's clips "
                               "folder, so queueing again continues the song instead of "
                               "starting over.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("SEGMENTS_DONE", "RENDER_PLAN", "SEGMENT_COUNT")
    FUNCTION = "render_segments"
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = ("Expands into one complete LTX 2.5 chain per segment, rendering "
                   "segments_per_queue segments per queue and skipping clips already on "
                   "disk - queue again to continue the song. Rendering everything in one "
                   "queue runs the machine out of memory. SEGMENTS_DONE fires after the "
                   "last clip of the batch is saved - wire it to the stitcher's "
                   "after_segments.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return _segments_cache_key(kwargs)

    def render_segments(self, force_rerun, model, clip, video_vae, audio_vae, images,
                        animation_prompts, storyboard_data, run_dir, width, height, fps,
                        seed, end_guide_strength, negative_prompt, max_segments=999,
                        img_compression=18, segments_per_queue=4, resume=True):
        from comfy_execution.graph_utils import GraphBuilder

        _stage_cleanup("LTXSegmentsRender")
        data = json.loads(storyboard_data)
        segments = data.get("segments", [])
        prompts = json.loads(animation_prompts)
        if not segments:
            raise ValueError("The storyboard has no segments - run Part 1 first.")
        if images.shape[0] < 2:
            raise ValueError(f"Need at least 2 cards for first/last-frame video, "
                             f"got {images.shape[0]}. Render the deck in Part 2 first.")
        deck = images.shape[0]
        wanted = segments[:max_segments]

        # ComfyUI holds EVERY node output of a queue alive until the queue ends.
        # One decoded 6s 1280x704 clip is ~1.5 GB of frames, so expanding a
        # thirty-segment song into one queue needs tens of gigabytes and dies
        # with an OOM before the first clip is written. Render a bounded batch,
        # skip what is already on disk, and let the user queue again.
        clips_dir = os.path.join(run_dir, "clips") if run_dir else ""
        pending, already = [], []
        for i, seg in enumerate(wanted):
            idx = int(seg.get("segment_index", i))
            clip_path = os.path.join(clips_dir, f"clip_{idx:03d}.mp4") if clips_dir else ""
            if resume and clip_path and os.path.isfile(clip_path):
                already.append(i)
            else:
                pending.append((i, seg))

        if resume:
            batch = pending[:max(1, int(segments_per_queue))]
        else:
            # 'Re-render every segment' leaves no on-disk progress to resume
            # from, so batching it would redo the same opening segments on every
            # queue and never reach the end. Do the whole song at once, and say
            # plainly that this is the setting that can run out of memory.
            batch = pending
            _safe_print("[LTXSegmentsRender] resume is OFF: rendering all "
                        f"{len(batch)} segment(s) in this single queue. This is the "
                        "setting that runs out of memory on a long song - turn resume "
                        "back on (and delete the clips you want redone) to render in "
                        "bounded batches instead.")
        remaining_after = len(pending) - len(batch)

        if not batch:
            done = "\n".join([
                "=" * 78,
                f" ALL {len(wanted)} SEGMENTS ARE ALREADY RENDERED",
                "=" * 78,
                f" Clips : {clips_dir or '(no run_dir)'}",
                " Nothing to render - the stitcher can build the final cut.",
                " To redo some, delete those clip_NNN.mp4 files and queue again;",
                " to redo all of them, switch 'resume' to 're-render every segment'.",
                "=" * 78])
            _safe_print(done)
            # Still send the board: finishing a song must not blank the table.
            finished = {"kind": "plan", "engine": "LTX 2.5",
                        "run": os.path.basename(run_dir or ""),
                        "total": len(wanted), "batch": 0, "done": len(already),
                        "remaining": 0,
                        "rows": _board_rows(wanted, prompts, run_dir, fps=fps, deck=deck,
                                            budget=VIDEO_PROMPT_BUDGETS["LTX 2.5"])}
            return {"ui": {"text": (done,), "gap_board": (json.dumps(finished),)},
                    "result": ("all segments already rendered", done, 0)}

        graph = GraphBuilder()
        negative = graph.node("CLIPTextEncode", id="neg", text=negative_prompt, clip=clip)
        sampler = graph.node("SamplerEulerAncestral", id="samp", eta=0.0, s_noise=1.0)
        sigmas = graph.node("ManualSigmas", id="sig", sigmas=self.DISTILLED_SIGMAS)

        plan_rows, chain, trimmed_count = [], None, 0
        # Every prompt's encode is collected here and forced to run BEFORE the
        # first segment samples - see the gather below the loop.
        positives, first_banner = [], None
        # `i` stays the segment's own index in the song, never its position in
        # this batch, so cards, seeds and clip numbers are identical no matter
        # how the song is split across queues.
        for i, seg in batch:
            a = min(i, deck - 1)
            b = min(i + 1, deck - 1)
            duration = float(seg.get("duration", MIN_SEGMENT_SEC))
            # LTX wants (frames - 1) divisible by 8; this keeps each clip within
            # 1/6 s of the segment so the video tracks the song, not whole seconds.
            frames = max(9, int(round(duration * fps / 8.0)) * 8 + 1)
            prompt = prompts[i] if i < len(prompts) else "Smooth cinematic motion"
            raw_len = len(prompt)
            prompt = _fit_prompt(prompt, VIDEO_PROMPT_BUDGETS["LTX 2.5"], keep_tail=2)
            if len(prompt) < raw_len:
                trimmed_count += 1

            # CONTINUITY: consecutive segments SHARE a card. Segment i ends on
            # card i+1 and segment i+1 starts on that exact same card i+1 -
            # a1->a2, a2->a3, a3->a4 - so every cut lands on a common frame.
            start_src = images[a:a + 1]
            start_desc = f"card {a}" + (f" (shared with segment {i})" if i > 0 else "")

            # The banner heads the segment's image path AND is chained behind the
            # previous segment's saver, so it fires exactly when this segment
            # starts: the canvas always names the segment in progress.
            banner_text = (
                "=" * 62 + "\n"
                f" NOW RENDERING   segment {i + 1} of {len(wanted)}\n"
                + "=" * 62 + "\n"
                f" Timecode : {_fmt_time(seg.get('start_time', 0))} -> "
                f"{_fmt_time(seg.get('end_time', 0))}   ({duration:.2f}s -> {frames} frames)\n"
                f" First    : {start_desc}\n"
                f" Last     : card {b}   seed {seed + i}\n"
                f" Lyric    : {_wrap(str(seg.get('lyrics', '-')), 50, 12)}\n"
                + "-" * 62 + "\n"
                " MOTION PROMPT sent to LTX 2.5:\n"
                + "\n".join("   " + ln for ln in _wrap(prompt, 56, 0).split("\n")) + "\n"
                + "=" * 62)
            banner_kwargs = dict(start_image=start_src, end_image=images[b:b + 1],
                                 info=banner_text,
                                 board=json.dumps({"kind": "now", "i": i}))
            if chain is not None:
                banner_kwargs["after"] = chain.out(0)
            banner = graph.node("SegmentNowRendering", id=f"now{i}", **banner_kwargs)

            first = graph.node("ImageScale", id=f"sA{i}", image=banner.out(0),
                               upscale_method="lanczos", width=width, height=height,
                               crop="center")
            last = graph.node("ImageScale", id=f"sB{i}", image=banner.out(1),
                              upscale_method="lanczos", width=width, height=height,
                              crop="center")
            # The two frames LTX is being fed, shown as this segment starts.
            graph.node("PreviewImage", id=f"prevA{i}", images=first.out(0))
            graph.node("PreviewImage", id=f"prevB{i}", images=last.out(0))
            prep_first = graph.node("LTXVPreprocess", id=f"pA{i}", image=first.out(0),
                                    img_compression=img_compression)
            prep_last = graph.node("LTXVPreprocess", id=f"pB{i}", image=last.out(0),
                                   img_compression=img_compression)

            if not positives:
                first_banner = banner
            positive = graph.node("CLIPTextEncode", id=f"pos{i}", text=prompt, clip=clip)
            positives.append(positive)
            cond = graph.node("LTXVConditioning", id=f"cond{i}",
                              positive=positive.out(0), negative=negative.out(0),
                              frame_rate=float(fps))
            v_latent = graph.node("EmptyLTXVLatentVideo", id=f"lat{i}",
                                  width=width, height=height, length=frames, batch_size=1)
            g_first = graph.node("LTXVAddGuide", id=f"gA{i}",
                                 positive=cond.out(0), negative=cond.out(1),
                                 vae=video_vae, latent=v_latent.out(0),
                                 image=prep_first.out(0), frame_idx=0, strength=0.7)
            g_last = graph.node("LTXVAddGuide", id=f"gB{i}",
                                positive=g_first.out(0), negative=g_first.out(1),
                                vae=video_vae, latent=g_first.out(2),
                                image=prep_last.out(0), frame_idx=-1,
                                strength=float(end_guide_strength))

            a_latent = graph.node("LTXVEmptyLatentAudio", id=f"aud{i}",
                                  frames_number=frames, frame_rate=fps, batch_size=1,
                                  audio_vae=audio_vae)
            av = graph.node("LTXVConcatAVLatent", id=f"av{i}",
                            video_latent=g_last.out(2), audio_latent=a_latent.out(0))

            noise = graph.node("RandomNoise", id=f"noise{i}", noise_seed=seed + i)
            guider = graph.node("LTXVDualCFGGuider", id=f"guide{i}", model=model,
                                positive=g_last.out(0), negative=g_last.out(1),
                                video_cfg=1.0, audio_cfg=1.0)
            sampled = graph.node("SamplerCustomAdvanced", id=f"ks{i}",
                                 noise=noise.out(0), guider=guider.out(0),
                                 sampler=sampler.out(0), sigmas=sigmas.out(0),
                                 latent_image=av.out(0))
            separate = graph.node("LTXVSeparateAVLatent", id=f"sep{i}",
                                  av_latent=sampled.out(1))
            cropped = graph.node("LTXVCropGuides", id=f"crop{i}",
                                 positive=g_last.out(0), negative=g_last.out(1),
                                 latent=separate.out(0))
            frames_out = graph.node("VAEDecodeTiled", id=f"dec{i}",
                                    samples=cropped.out(2), vae=video_vae,
                                    tile_size=768, overlap=64,
                                    temporal_size=4096, temporal_overlap=64)
            audio_out = graph.node("LTXVAudioVAEDecode", id=f"adec{i}",
                                   samples=separate.out(1), audio_vae=audio_vae)
            video = graph.node("CreateVideo", id=f"vid{i}",
                               images=frames_out.out(0), audio=audio_out.out(0),
                               fps=float(fps))

            # Saved into the run under its segment number as soon as it exists.
            # Chaining each saver behind the previous one renders the song in
            # order, and makes SEGMENTS_DONE depend on every clip.
            saver_kwargs = dict(force_rerun=False, video=video.out(0), run_dir=run_dir,
                                segment_index=seg.get("segment_index", i),
                                storyboard_data=storyboard_data,
                                per_queue=max(1, int(segments_per_queue)) if resume else 0)
            if chain is not None:
                saver_kwargs["after"] = chain.out(0)
            chain = graph.node("SegmentVideoSaver", id=f"save{i}", **saver_kwargs)

            plan_rows.append(
                f" seg {i + 1:>3}/{len(wanted)}  {_fmt_time(seg.get('start_time', 0))} -> "
                f"{_fmt_time(seg.get('end_time', 0))}  {duration:5.2f}s -> {frames} frames  "
                f"cards {a}->{b}  seed {seed + i}\n"
                f"     {_wrap(prompt, 70, 5)}")

        # ------------------------------------------------------------------
        # Encode every prompt BEFORE the first segment samples.
        #
        # A segment's chain is encode -> sample: the encode needs Gemma (the
        # 14.6 GB LTX text encoder) and the sample needs the 22B transformer.
        # Both cannot sit in 24 GB together, so running the segments as
        # encode/sample/encode/sample makes the two models swap once PER
        # SEGMENT - 64 full model loads for a 32-segment song. Every swap
        # re-streams weights from disk, and it is the text-encoder load that
        # eventually fails.
        #
        # The official LTX 2.5 first/last-frame workflow encodes once because
        # it renders one clip. We render N clips with N prompts, so the
        # equivalent is: encode all N prompts while Gemma is resident, then
        # load the transformer once and sample every segment.
        #
        # ConditioningCombine only concatenates lists, so this gather is free.
        # Its result is fed to the first banner's `after` input, which exists
        # purely for sequencing and is ignored by the node - it just makes the
        # whole batch of encodes a prerequisite of the first sample. The
        # negative seeds the chain so it is encoded in the same pass.
        # ------------------------------------------------------------------
        if first_banner is not None and positives:
            gather = negative
            for k, pos in enumerate(positives):
                gather = graph.node("ConditioningCombine", id=f"warm{k}",
                                    conditioning_1=gather.out(0),
                                    conditioning_2=pos.out(0))
            first_banner.set_input("after", gather.out(0))
            _safe_print(f"[LTXSegmentsRender] Encoding {len(positives)} prompt(s) + the "
                        f"negative in one pass before sampling, so Gemma and the "
                        f"transformer swap once instead of once per segment.")

        # ------------------------------------------------------------------
        # Load and show the frames BEFORE anything renders.
        #
        # Each segment's own preview sits inside that segment's chain, so it
        # only appears as that segment starts - by which point you are already
        # committed. This batches the first/last card of every segment in this
        # queue into one preview, in order (seg1-first, seg1-last, seg2-first,
        # ...), and makes it a prerequisite of the first banner, so the whole
        # set is on screen before the first frame is sampled. Confirming that
        # the right pairs are being animated is then possible while there is
        # still time to stop the queue.
        #
        # It is wired into the banner's `frames_ready` input rather than
        # `after` because `after` already carries the text-encode gather, and
        # both inputs exist purely for sequencing - the node ignores them.
        # ------------------------------------------------------------------
        pair_preview, held, pair_count = None, None, 0
        for i, seg in batch:
            a = min(i, deck - 1)
            b = min(i + 1, deck - 1)
            for frame in (images[a:a + 1], images[b:b + 1]):
                if pair_preview is None and held is None:
                    held = frame                      # ImageBatch needs a pair
                elif pair_preview is None:
                    pair_preview = graph.node("ImageBatch", id=f"pair{pair_count}",
                                              image1=held, image2=frame)
                    held = None
                    pair_count += 1
                else:
                    pair_preview = graph.node("ImageBatch", id=f"pair{pair_count}",
                                              image1=pair_preview.out(0), image2=frame)
                    pair_count += 1
        if pair_preview is not None and first_banner is not None:
            graph.node("PreviewImage", id="pairs_all", images=pair_preview.out(0))
            first_banner.set_input("frames_ready", pair_preview.out(0))
            _safe_print(f"[LTXSegmentsRender] Loading the {len(batch) * 2} first/last frame(s) "
                        f"for this queue up front - they are previewed before the first "
                        f"segment is sampled.")

        bar = "=" * 78
        plan = "\n".join(
            _json_source_rows(data, raw=storyboard_data,
                              title="SEGMENT RENDER - the JSON this video is cut to")
            + [f" Deck: {deck} cards   Canvas: {width}x{height}@{fps}fps   "
               f"end guide {end_guide_strength:.2f}",
               f" Clips -> {os.path.join(run_dir, 'clips')}",
               bar,
               f" RENDERING {len(batch)} SEGMENT(S) IN THIS QUEUE"
               f"   (song has {len(wanted)}; {len(already)} already on disk, "
               f"{remaining_after} still to go)",
               bar]
            + plan_rows
            + ["-" * 78,
               " Each clip is saved the moment it finishes."])
        if trimmed_count:
            plan += (f"\n {trimmed_count} motion prompt(s) were trimmed to LTX 2.5's budget "
                     f"(~{VIDEO_PROMPT_BUDGETS['LTX 2.5']} chars), keeping camera,\n"
                     f" action and the continuity clause; decorative style sentences were dropped.")
        if remaining_after:
            more = -(-remaining_after // max(1, int(segments_per_queue)))   # ceil
            plan += (f"\n {remaining_after} segment(s) still to render after this queue.\n"
                     f" Set the Queue batch count to {more} and queue once - each run picks up "
                     f"where the last\n one stopped, because finished clips are skipped.\n"
                     f" segments_per_queue={segments_per_queue} is what bounds the memory a single "
                     f"queue needs: lower it\n if you still hit an out-of-memory error, raise it "
                     f"only if you have memory to spare.")
        else:
            plan += "\n This queue finishes the song - the stitcher can build the final cut next."
        if len(segments) > len(wanted):
            plan += f"\n max_segments capped this at {len(wanted)} of {len(segments)} segments."
        plan += "\n" + bar
        _safe_print(plan)

        # The board the canvas draws: EVERY segment of the song, not just the
        # ones this queue renders, so you can see the whole timeline, which
        # segments are already on disk, which this queue is about to do, and
        # where the render currently is. Each row carries the two cards being
        # animated, the prompt driving them, and the finished clip once it
        # exists - the banner and saver messages below update rows in place.
        rows = _board_rows(wanted, prompts, run_dir, fps=fps, deck=deck,
                           queued={i for i, _ in batch},
                           budget=VIDEO_PROMPT_BUDGETS["LTX 2.5"])
        board = {"kind": "plan", "engine": "LTX 2.5",
                 "run": os.path.basename(run_dir or ""),
                 "total": len(wanted), "batch": len(batch),
                 "done": len(already), "remaining": remaining_after, "rows": rows}

        return {"ui": {"text": (plan,), "gap_board": (json.dumps(board),)},
                "expand": graph.finalize(),
                "result": (chain.out(0), plan, len(batch))}


def _minimax_frames(duration, fps):
    """
    Frame count MiniMax H3 will accept for a clip of `duration` seconds.

    Valid lengths are 5, 22, 39, ... - five plus a multiple of seventeen.
    Feeding anything else produces a shape error deep in the sampler, so the
    rounding has to happen here rather than being left to the user.

    The reference graph always rounds UP to the next valid length. At 24fps that
    quantum is 0.7s, so rounding up alone adds up to +0.67s of video to every
    segment - which on a thirty-segment song is a video that runs a quarter of a
    minute longer than the song it is cut to. We round to the NEAREST valid
    length instead, which centres the error and roughly halves it. The stitcher
    trims each clip to its true segment length on top of this.
    """
    n = max(5, int(round(float(duration) * int(fps))))
    steps = max(0, int(round((n - 5) / 17.0)))
    return 5 + 17 * steps


class MiniMaxSegmentsRender(ForceRerunMixin):
    """
    Renders the song's video segments with MiniMax H3, `segments_per_queue` at
    a time, the same way LTXSegmentsRender does it with LTX 2.5 - and for the
    same reason: a whole song expanded into one queue exhausts memory.

    MiniMax H3 takes the first and last frame directly on one node and returns
    a joint audio-video latent, so each segment's chain is much shorter than the
    LTX one: scale both cards -> MiniMaxH3ImageToVideo with the segment's motion
    prompt -> BasicGuider -> SamplerCustomAdvanced -> decode video and audio off
    the same latent -> CreateVideo. Node for node this mirrors ComfyUI's own
    `video_minimax_h3_i2v` template.

    There is deliberately no negative prompt: MiniMax H3 is guidance-free, the
    conditioning node emits only a positive, and BasicGuider takes no negative.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_rerun": FORCE_RERUN_INPUT,
                "model": ("MODEL", {"tooltip": "MiniMax H3 transformer, e.g. "
                                               "minimax_h3_fl2va_*.safetensors (UNETLoader)."}),
                "clip": ("CLIP", {"tooltip": "MiniMax H3 text encoder "
                                             "(CLIPLoader, type 'minimax')."}),
                "video_vae": ("VAE", {"tooltip": "minimax_h3_video_vae_*.safetensors"}),
                "audio_vae": ("VAE", {"tooltip": "minimax_h3_audio_vae_*.safetensors"}),
                "images": ("IMAGE", {"tooltip": "The full card deck from the card loader."}),
                "animation_prompts": ("STRING", {"forceInput": True}),
                "storyboard_data": ("STRING", {"forceInput": True}),
                "run_dir": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 704, "min": 256, "max": 4096, "step": 32}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": False,
                                 "tooltip": "Segment i renders with seed + i, so one seed "
                                            "drives the whole song reproducibly."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100,
                                  "tooltip": "20 is the reference setting for MiniMax H3."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS if comfy else ["res_multistep"],
                                 {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS if comfy else ["simple"],
                              {"default": "simple"}),
                "max_segments": ("INT", {"default": 999, "min": 1, "max": 999,
                                         "tooltip": "Safety cap. Leave at 999 to render "
                                                    "the whole song."}),
            },
            "optional": {
                "segments_per_queue": ("INT", {
                    "default": 4, "min": 1, "max": 999,
                    "tooltip": "How many segments to render in ONE queue. ComfyUI keeps every "
                               "node's output alive until the queue ends, and a single decoded "
                               "clip is over a gigabyte, so expanding a whole song at once "
                               "exhausts memory and OOMs. Render a few at a time and queue "
                               "again - finished clips are skipped.",
                }),
                "resume": ("BOOLEAN", {
                    "default": True,
                    "label_on": "skip clips already rendered",
                    "label_off": "re-render every segment",
                    "tooltip": "Skips any segment whose clip is already in the run's clips "
                               "folder, so queueing again continues the song instead of "
                               "starting over.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("SEGMENTS_DONE", "RENDER_PLAN", "SEGMENT_COUNT")
    FUNCTION = "render_segments"
    CATEGORY = "Geekatplay/VideoClipMaker"
    DESCRIPTION = ("Expands into one complete MiniMax H3 chain per segment, rendering "
                   "segments_per_queue segments per queue and skipping clips already on "
                   "disk - queue again to continue the song. SEGMENTS_DONE fires after "
                   "the last clip of the batch is saved - wire it to the stitcher's "
                   "after_segments.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return _segments_cache_key(kwargs)

    def render_segments(self, force_rerun, model, clip, video_vae, audio_vae, images,
                        animation_prompts, storyboard_data, run_dir, width, height, fps,
                        seed, steps, sampler_name, scheduler, max_segments=999,
                        segments_per_queue=4, resume=True):
        from comfy_execution.graph_utils import GraphBuilder

        _stage_cleanup("MiniMaxSegmentsRender")
        data = json.loads(storyboard_data)
        segments = data.get("segments", [])
        prompts = json.loads(animation_prompts)
        if not segments:
            raise ValueError("The storyboard has no segments - run Part 1 first.")
        if images.shape[0] < 2:
            raise ValueError(f"Need at least 2 cards for first/last-frame video, "
                             f"got {images.shape[0]}. Render the deck in Part 2 first.")
        deck = images.shape[0]
        wanted = segments[:max_segments]

        # See LTXSegmentsRender: one queue holds every decoded clip alive at
        # once, so a whole song in one queue runs the machine out of memory.
        clips_dir = os.path.join(run_dir, "clips") if run_dir else ""
        pending, already = [], []
        for i, seg in enumerate(wanted):
            idx = int(seg.get("segment_index", i))
            clip_path = os.path.join(clips_dir, f"clip_{idx:03d}.mp4") if clips_dir else ""
            if resume and clip_path and os.path.isfile(clip_path):
                already.append(i)
            else:
                pending.append((i, seg))

        if resume:
            batch = pending[:max(1, int(segments_per_queue))]
        else:
            # See LTXSegmentsRender: batching a full re-render would never finish.
            batch = pending
            _safe_print("[MiniMaxSegmentsRender] resume is OFF: rendering all "
                        f"{len(batch)} segment(s) in this single queue - this is the "
                        "setting that runs out of memory on a long song.")
        remaining_after = len(pending) - len(batch)

        if not batch:
            done = "\n".join([
                "=" * 78,
                f" ALL {len(wanted)} SEGMENTS ARE ALREADY RENDERED",
                "=" * 78,
                f" Clips : {clips_dir or '(no run_dir)'}",
                " Nothing to render - the stitcher can build the final cut.",
                " To redo some, delete those clip_NNN.mp4 files and queue again;",
                " to redo all of them, switch 'resume' to 're-render every segment'.",
                "=" * 78])
            _safe_print(done)
            return {"ui": {"text": (done,)}, "result": ("all segments already rendered",
                                                        done, 0)}

        graph = GraphBuilder()
        sampler = graph.node("KSamplerSelect", id="samp", sampler_name=sampler_name)
        sigmas = graph.node("BasicScheduler", id="sig", model=model, scheduler=scheduler,
                            steps=steps, denoise=1.0)

        plan_rows, chain, trimmed_count = [], None, 0
        # `i` is the segment's index in the SONG, not in this batch, so cards,
        # seeds and clip numbers do not shift when the song is split up.
        for i, seg in batch:
            a = min(i, deck - 1)
            b = min(i + 1, deck - 1)
            duration = float(seg.get("duration", MIN_SEGMENT_SEC))
            frames = _minimax_frames(duration, fps)
            prompt = prompts[i] if i < len(prompts) else "Smooth cinematic motion"
            raw_len = len(prompt)
            prompt = _fit_prompt(prompt, VIDEO_PROMPT_BUDGETS["MiniMax H3"], keep_tail=2)
            if len(prompt) < raw_len:
                trimmed_count += 1

            # CONTINUITY: consecutive segments SHARE a card - segment i ends on
            # card i+1 and segment i+1 opens on that same card - so every cut
            # lands on a frame both clips have in common.
            start_src = images[a:a + 1]
            start_desc = f"card {a}" + (f" (shared with segment {i})" if i > 0 else "")

            banner_text = (
                "=" * 62 + "\n"
                f" NOW RENDERING   segment {i + 1} of {len(wanted)}\n"
                + "=" * 62 + "\n"
                f" Timecode : {_fmt_time(seg.get('start_time', 0))} -> "
                f"{_fmt_time(seg.get('end_time', 0))}   ({duration:.2f}s -> {frames} frames)\n"
                f" First    : {start_desc}\n"
                f" Last     : card {b}   seed {seed + i}\n"
                f" Lyric    : {_wrap(str(seg.get('lyrics', '-')), 50, 12)}\n"
                + "-" * 62 + "\n"
                " MOTION PROMPT sent to MiniMax H3:\n"
                + "\n".join("   " + ln for ln in _wrap(prompt, 56, 0).split("\n")) + "\n"
                + "=" * 62)
            banner_kwargs = dict(start_image=start_src, end_image=images[b:b + 1],
                                 info=banner_text)
            if chain is not None:
                banner_kwargs["after"] = chain.out(0)
            banner = graph.node("SegmentNowRendering", id=f"now{i}", **banner_kwargs)

            first = graph.node("ImageScale", id=f"sA{i}", image=banner.out(0),
                               upscale_method="lanczos", width=width, height=height,
                               crop="center")
            last = graph.node("ImageScale", id=f"sB{i}", image=banner.out(1),
                              upscale_method="lanczos", width=width, height=height,
                              crop="center")
            graph.node("PreviewImage", id=f"prevA{i}", images=first.out(0))
            graph.node("PreviewImage", id=f"prevB{i}", images=last.out(0))

            # One node carries the prompt, both keyframes and the empty AV latent.
            cond = graph.node("MiniMaxH3ImageToVideo", id=f"cond{i}",
                              clip=clip, vae=video_vae, prompt=prompt,
                              width=width, height=height, length=frames,
                              first_frame=first.out(0), last_frame=last.out(0))

            noise = graph.node("RandomNoise", id=f"noise{i}", noise_seed=seed + i)
            guider = graph.node("BasicGuider", id=f"guide{i}", model=model,
                                conditioning=cond.out(0))
            sampled = graph.node("SamplerCustomAdvanced", id=f"ks{i}",
                                 noise=noise.out(0), guider=guider.out(0),
                                 sampler=sampler.out(0), sigmas=sigmas.out(0),
                                 latent_image=cond.out(1))

            # One joint latent decodes twice: picture through the video VAE,
            # the model's own generated audio through the audio VAE.
            frames_out = graph.node("VAEDecode", id=f"dec{i}",
                                    samples=sampled.out(0), vae=video_vae)
            audio_out = graph.node("VAEDecodeAudio", id=f"adec{i}",
                                   samples=sampled.out(0), vae=audio_vae)
            video = graph.node("CreateVideo", id=f"vid{i}",
                               images=frames_out.out(0), audio=audio_out.out(0),
                               fps=float(fps))

            saver_kwargs = dict(force_rerun=False, video=video.out(0), run_dir=run_dir,
                                segment_index=seg.get("segment_index", i),
                                storyboard_data=storyboard_data,
                                per_queue=max(1, int(segments_per_queue)) if resume else 0)
            if chain is not None:
                saver_kwargs["after"] = chain.out(0)
            chain = graph.node("SegmentVideoSaver", id=f"save{i}", **saver_kwargs)

            plan_rows.append(
                f" seg {i + 1:>3}/{len(wanted)}  {_fmt_time(seg.get('start_time', 0))} -> "
                f"{_fmt_time(seg.get('end_time', 0))}  {duration:5.2f}s -> {frames} frames  "
                f"cards {a}->{b}  seed {seed + i}\n"
                f"     {_wrap(prompt, 70, 5)}")

        bar = "=" * 78
        plan = "\n".join(
            _json_source_rows(data, raw=storyboard_data,
                              title="SEGMENT RENDER (MiniMax H3) - the JSON this video is cut to")
            + [f" Deck: {deck} cards   Canvas: {width}x{height}@{fps}fps   "
               f"{steps} steps, {sampler_name}/{scheduler}",
               f" Clips -> {os.path.join(run_dir, 'clips')}",
               bar,
               f" RENDERING {len(batch)} SEGMENT(S) IN THIS QUEUE"
               f"   (song has {len(wanted)}; {len(already)} already on disk, "
               f"{remaining_after} still to go)",
               bar]
            + plan_rows
            + ["-" * 78,
               " Each clip is saved the moment it finishes."])
        if trimmed_count:
            plan += (f"\n {trimmed_count} motion prompt(s) were trimmed to MiniMax H3's budget "
                     f"(~{VIDEO_PROMPT_BUDGETS['MiniMax H3']} chars), keeping camera,\n"
                     f" action and the continuity clause; decorative style sentences were dropped.")
        if remaining_after:
            more = -(-remaining_after // max(1, int(segments_per_queue)))   # ceil
            plan += (f"\n {remaining_after} segment(s) still to render after this queue.\n"
                     f" Set the Queue batch count to {more} and queue once - finished clips "
                     f"are skipped.")
        else:
            plan += "\n This queue finishes the song - the stitcher can build the final cut next."
        if len(segments) > len(wanted):
            plan += f"\n max_segments capped this at {len(wanted)} of {len(segments)} segments."
        plan += "\n" + bar
        _safe_print(plan)

        return {"ui": {"text": (plan,)},
                "expand": graph.finalize(),
                "result": (chain.out(0), plan, len(batch))}


class SegmentVideoSaver(ForceRerunMixin):
    """
    Writes the clip LTX 2.5 just rendered into this run's folder, named after the
    segment it belongs to: <run>/clips/clip_000.mp4.

    ComfyUI's SaveVideo writes auto-numbered files into one shared output folder,
    so a finished clip carries no record of which segment - or which project - it
    came from. A stitcher that simply lists that folder will happily pick up
    clips left over from yesterday and report a complete movie. Naming the file
    after the segment index makes each run self-contained, and re-rendering one
    segment overwrites exactly that segment and nothing else.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_rerun": FORCE_RERUN_INPUT,
                "video": ("VIDEO", ),
                "run_dir": ("STRING", {"forceInput": True}),
                "segment_index": ("INT", {"forceInput": True}),
            },
            "optional": {
                "storyboard_data": ("STRING", {"forceInput": True}),
                # LTXSegmentsRender chains savers through this, so the segments
                # render in order and the last CLIP_PATH depends on every clip.
                "after": (ANY, ),
                # How many segments each queue renders, so the advice below can
                # say how many QUEUES remain instead of a misleading number.
                "per_queue": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "VIDEO", "STRING")
    RETURN_NAMES = ("CLIP_PATH", "VIDEO", "REPORT")
    FUNCTION = "save_segment"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"

    def save_segment(self, video, run_dir, segment_index, storyboard_data=None,
                     force_rerun=False, after=None, per_queue=0):
        fallback = folder_paths.get_output_directory() if folder_paths else "."
        target_dir = run_dir if (run_dir and os.path.isdir(run_dir)) else fallback
        clips_dir = os.path.join(target_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        idx = int(segment_index)
        path = os.path.join(clips_dir, f"clip_{idx:03d}.mp4")
        if os.path.isfile(path):
            os.remove(path)          # re-render replaces this segment in place

        try:
            if ComfyVideoTypes is not None:
                video.save_to(path,
                              format=ComfyVideoTypes.VideoContainer.MP4,
                              codec=ComfyVideoTypes.VideoCodec.H264)
            else:
                video.save_to(path)
        except Exception as exc:
            report = f"Could not write clip {idx}: {exc}"
            _safe_print(f"[SegmentVideoSaver] {report}")
            return {"ui": {"text": (report,)}, "result": ("", video, report)}

        total = 0
        if storyboard_data:
            try:
                total = len(json.loads(storyboard_data).get("segments", []))
            except Exception:
                total = 0

        have = sorted(f for f in os.listdir(clips_dir)
                      if f.startswith("clip_") and f.endswith(".mp4"))
        bar = "=" * 78
        rows = [bar, f" CLIP SAVED   segment {idx + 1}" + (f" of {total}" if total else ""), bar,
                f" File   : {path}",
                f" In run : {len(have)}" + (f" of {total} clips present" if total else " clip(s)")]
        if total:
            missing = [i for i in range(total)
                       if f"clip_{i:03d}.mp4" not in have]
            if missing:
                shown = ", ".join(str(m) for m in missing[:12])
                if len(missing) > 12:
                    shown += ", ..."
                rows.append(f" MISSING: segment(s) {shown}")
                if per_queue and int(per_queue) > 0:
                    queues = -(-len(missing) // int(per_queue))   # ceil
                    rows.append(f" {len(missing)} segment(s) left = {queues} more queue(s) at "
                                f"{int(per_queue)} per queue. Set the Queue batch count to "
                                f"{queues} and queue once.")
                else:
                    rows.append(f" {len(missing)} segment(s) left - queue Part 3 again until "
                                f"none remain (finished clips are skipped).")
            else:
                rows.append(" All segments rendered - the stitcher can build the final cut.")
        rows.append(bar)
        report = "\n".join(rows)
        _safe_print(report)
        # Hand the finished clip to the board so its row can play it back.
        done_board = json.dumps({"kind": "done", "i": idx, "clip": _view_ref(path)})

        # Preview the clip on the canvas. The run folder lives under ComfyUI's
        # output directory, so the frontend can serve it from there directly -
        # no need for a second copy in a shared folder.
        ui = {"text": (report,), "gap_board": (done_board,)}
        try:
            out_root = folder_paths.get_output_directory() if folder_paths else None
            if out_root:
                rel = os.path.relpath(clips_dir, out_root)
                if not rel.startswith(".."):
                    ui["images"] = [{"filename": os.path.basename(path),
                                     "subfolder": rel.replace("\\", "/"),
                                     "type": "output"}]
                    ui["animated"] = (True,)
        except Exception:
            pass
        return {"ui": ui, "result": (path, video, report)}


class VideoSegmentStitcher(ForceRerunMixin):
    """
    Stitches generated segment videos into a continuous music video and attaches original song audio track.
    Cleans video generation models from VRAM after rendering is complete.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_rerun": FORCE_RERUN_INPUT,
                "audio_path": ("STRING", {"forceInput": True}),
                "segment_folder": ("STRING", {"default": "video"}),
                "filename_filter": ("STRING", {"default": "LTX-2.5_i2v"}),
                "output_filename": ("STRING", {"default": "final_music_video.mp4"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "audio_per_segment": ("BOOLEAN", {
                    "default": True,
                    "label_on": "each segment keeps its own slice of the song",
                    "label_off": "segments stay silent",
                }),
            },
            "optional": {
                "storyboard_data": ("STRING", {"forceInput": True}),
                "run_dir": ("STRING", {"forceInput": True}),
                # Wire the video step's output here so stitching only happens after
                # segments exist. Without it this node sits upstream of both approval
                # gates and would run - and "finish" the movie - on the very first queue.
                "after_segments": (ANY, ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("FINAL_VIDEO_PATH", "SEGMENTS_DIR", "REPORT")
    FUNCTION = "stitch_videos"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay/VideoClipMaker"

    @staticmethod
    def _run_ffmpeg(cmd):
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0, (result.stderr or "").strip()

    def stitch_videos(self, audio_path, segment_folder, filename_filter, output_filename,
                      fps=24, audio_per_segment=True, force_rerun=False,
                      after_segments=None, storyboard_data=None, run_dir=None):
        # Unload LTX 2.5 models from VRAM
        if mm is not None:
            mm.unload_all_models()
            mm.soft_empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        output_dir = folder_paths.get_output_directory() if folder_paths else "."

        # Everything for a run belongs with that run.
        target_dir = run_dir if (run_dir and os.path.isdir(run_dir)) else output_dir
        segments_dir = os.path.join(target_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)
        final_output_path = os.path.join(target_dir, output_filename)

        segments = []
        if storyboard_data:
            try:
                segments = json.loads(storyboard_data).get("segments", [])
            except Exception:
                segments = []

        have_song = bool(audio_path) and os.path.isfile(audio_path)
        if not have_song:
            _safe_print(f"[VideoSegmentStitcher] WARNING: the original song was not found at "
                        f"'{audio_path}'. The video will be built without audio. Re-run Part 1 "
                        "so the song is stored with the run.")

        expected = len(segments)

        # Clips written by SegmentVideoSaver carry their segment number in the
        # filename, so a run only ever joins its own clips. The alternative -
        # listing a shared output folder - silently mixes in clips from other
        # projects and from earlier attempts, and then reports a complete movie.
        clips_dir = os.path.join(target_dir, "clips")
        indexed = {}
        if os.path.isdir(clips_dir):
            for name in os.listdir(clips_dir):
                match = re.fullmatch(r"clip_(\d+)\.(?:mp4|mkv|webm)", name, re.IGNORECASE)
                if match:
                    indexed[int(match.group(1))] = os.path.join(clips_dir, name)

        missing = []
        if indexed:
            order = sorted(indexed)
            video_files = [indexed[i] for i in order]
            source = clips_dir
            missing = [i for i in range(expected) if i not in indexed] if expected else []
            seg_for_clip = order
        else:
            # Nothing from this run: fall back to the old shared-folder scan, but
            # say so, because those files cannot be attributed to any segment.
            if not segment_folder.strip():
                segment_folder = "video"
            if not os.path.isabs(segment_folder):
                segment_folder = os.path.join(output_dir, segment_folder)
            os.makedirs(segment_folder, exist_ok=True)

            name_filter = filename_filter.strip()
            video_files = sorted([
                os.path.join(segment_folder, f) for f in os.listdir(segment_folder)
                if f.lower().endswith(('.mp4', '.mkv', '.webm'))
                and (not name_filter or f.startswith(name_filter))
                and f != output_filename
            ])
            source = segment_folder
            seg_for_clip = list(range(len(video_files)))
            if video_files:
                _safe_print("[VideoSegmentStitcher] " + "!" * 60)
                _safe_print(f"[VideoSegmentStitcher] No clips/ folder in this run, so falling "
                            f"back to listing {segment_folder}.")
                _safe_print("[VideoSegmentStitcher] Those files are NOT tied to this project - "
                            "they may be left over from other runs or other days. Wire the "
                            "'Save this segment's clip' node in Part 3 so every clip is stored "
                            "with its segment number.")
                _safe_print("[VideoSegmentStitcher] " + "!" * 60)

        if not video_files:
            report = (f"No rendered clips found in {source}.\n"
                      "Render the video segments first (Part 3), then queue this again.")
            _safe_print(f"[VideoSegmentStitcher] {report}")
            return {"ui": {"text": (report,)},
                    "result": (final_output_path, segments_dir, report)}

        if missing:
            shown = ", ".join(str(m) for m in missing[:12]) + (", ..." if len(missing) > 12 else "")
            _safe_print("[VideoSegmentStitcher] " + "!" * 60)
            _safe_print(f"[VideoSegmentStitcher] Segment(s) {shown} have not been rendered. "
                        f"The final video will be {len(video_files)} of {expected} clips long, "
                        f"not the whole song.")
            _safe_print(f"[VideoSegmentStitcher] Render the rest first: set the Queue batch "
                        f"count to {expected} and queue once.")
            _safe_print("[VideoSegmentStitcher] " + "!" * 60)

        # ---- 1. every segment, with its own slice of the original song ----
        kept = []
        for position, clip in enumerate(video_files):
            # Which segment this clip actually is, not merely where it sits in the list.
            index = seg_for_clip[position] if position < len(seg_for_clip) else position
            destination = os.path.join(segments_dir, f"segment_{index:03d}.mp4")
            if have_song and audio_per_segment and index < len(segments):
                seg = segments[index]
                ok, err = self._run_ffmpeg([
                    "ffmpeg", "-y",
                    "-i", clip,
                    "-ss", f"{float(seg['start_time']):.3f}",
                    "-t", f"{float(seg['duration']):.3f}",
                    "-i", audio_path,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-shortest",
                    destination,
                ])
                if not ok:
                    _safe_print(f"[VideoSegmentStitcher] segment {index} audio mux failed: {err[:200]}")
                    shutil.copy2(clip, destination)
            else:
                shutil.copy2(clip, destination)
            kept.append(destination)

        # ---- 2. the whole video, with the full original song ----
        # Concatenate the SEGMENTS WE JUST BUILT, not the raw clips. Step 1 muxes
        # each clip against its own slice of the song with -shortest, which trims
        # the video back to the segment's true length; a video model can only
        # render whole frames, and some quantise hard (MiniMax H3 to 17 frames,
        # ~0.7s at 24fps), so a raw clip always runs a little long. Concatenating
        # the raw clips lets that overshoot accumulate - every segment pushes the
        # picture further ahead of the song, and by the last chorus the video is
        # visibly off the beat.
        concat_list_path = os.path.join(segments_dir, "concat_list.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for vfile in kept:
                clean_path = vfile.replace("\\", "/")
                f.write(f"file '{clean_path}'\n")

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path]
        if have_song:
            cmd += ["-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-shortest"]
        else:
            cmd += ["-map", "0:v:0", "-c:v", "copy"]
        cmd.append(final_output_path)

        ok, err = self._run_ffmpeg(cmd)

        bar = "=" * 78
        rows = [bar, " FINAL CUT", bar,
                f" Clips joined : {len(video_files)}"
                + (f" of {expected} segments" if expected else "")
                + ("   *** INCOMPLETE - render the rest ***"
                   if expected and len(video_files) < expected else ""),
                f" Clips from   : {source}",
                f" Segments     : {segments_dir}"
                + ("  (each with its own slice of the song)" if have_song and audio_per_segment else ""),
                f" Soundtrack   : {audio_path if have_song else 'MISSING - video has no audio'}"]
        if missing:
            shown = ", ".join(str(m) for m in missing[:12]) + (", ..." if len(missing) > 12 else "")
            rows.append(f" NOT RENDERED : segment(s) {shown}")
            rows.append(f"                {len(missing)} left; each queue renders a batch "
                        f"(segments_per_queue, default 4),")
            rows.append(f"                so set the Queue batch count to "
                        f"~{-(-len(missing) // 4)} and queue once")
        if not indexed:
            rows.append(" WARNING      : these clips came from a shared output folder and are")
            rows.append("                NOT tied to this project - wire the clip saver in Part 3")
        if ok:
            rows.append(f" Final video  : {final_output_path}")
        else:
            rows.append(" FFmpeg failed: " + (err[-400:] if err else "unknown error"))
            rows.append(" Is ffmpeg installed and on PATH?")
        rows += [_gap_credit(), bar]
        report = "\n".join(rows)
        _safe_print(f"[VideoSegmentStitcher]\n{report}")

        return {"ui": {"text": (report,)},
                "result": (final_output_path, segments_dir, report)}
