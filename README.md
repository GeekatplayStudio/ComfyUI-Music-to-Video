<div align="center">

# ComfyUI-Music-to-Video

### Geekatplay Studio &mdash; Vladimir Chopine

**Turn a song into an AI music video, inside ComfyUI**

</div>

---

## Thank you for your support!

If this node pack is useful to you, please **give the project a star on GitHub** &mdash;
it genuinely helps other people find it.

[**Star on GitHub**](https://github.com/GeekatplayStudio/ComfyUI-Music-to-Video)

And please subscribe on YouTube for tutorials, breakdowns and new tools:

| Channel | |
| :--- | :--- |
| [**youtube.com/@geekatplay**](https://www.youtube.com/@geekatplay) | English |
| [**youtube.com/@geekatplay-ru**](https://www.youtube.com/@geekatplay-ru) | Russian |
| [**youtube.com/@v-code-studio**](https://www.youtube.com/@v-code-studio) | **V-Code Studio** &mdash; my new channel |

---

## What it does

Give it a song. It transcribes the lyrics, maps them to a timeline **cut on the beat**,
works out who and where the song is about, renders reference sheets so the cast and the
world stay consistent, renders a storyboard card for every cut, animates each pair of
cards with **LTX 2.5 or MiniMax H3**, and joins everything back together with your
original soundtrack.

**Every creative decision is derived from your song.** The subject, the setting, the
mood, the palette, the camera movement and the pacing all come from the lyrics and the
music analysis. Nothing is templated in &mdash; the fields where you *can* type your own
direction are optional overrides, and they ship empty.

```
song ──► lyrics + beat-aware timeline ──► reference sheets ──► storyboard cards ──► LTX 2.5 clips ──► final video
              ⏸ STOP 1                                            ⏸ STOP 2
```

---

## Quick start

1. **Install** (see below), then restart ComfyUI.
2. Open **`example_workflows/music_video_ALL_IN_ONE.json`**.
3. Load your song in the **Your song** node and press **Queue**.
4. Read the panels &mdash; timeline, full lyrics, references, every card prompt. Adjust
   `style_preset` / `prompt_approach` / segment timings if you want. Nothing has been
   written yet and no model has loaded.
5. Flip **`approve_timeline`** &rarr; Queue. References and **all cards** render.
6. Flip **`approve_cards`** &rarr; Queue. Video renders in memory-safe batches of four
   segments by default. Queue again until the status reports that every clip is present;
   completed clips are skipped and the final cut is rebuilt from the available clips.

Two stops, followed by as many video batches as the song needs. Finished work stays on
disk, so each video queue continues where the previous one ended. You can also set
ComfyUI's Queue batch count to the number reported by the status panel and queue once.

---

## Workflows Included (`example_workflows/`)

Two axes: where the **song** comes from (a file you load, or MiniMax Music 3
generating one from a caption), and which **video** backend renders the segments.

### All-In-One Workflows
| Workflow File | Song source | Video | Best For |
| :--- | :--- | :--- | :--- |
| **`music_video_ALL_IN_ONE.json`** | audio file | LTX 2.5 | **Start here.** Everything in one graph: two stops, then resumable video batches. |
| **`music_video_ALL_IN_ONE_LTX.json`** | audio file | LTX 2.5 | The LTX-named copy of the flagship, for keeping an LTX-only set together. |
| **`music_video_ALL_IN_ONE_MINIMAX.json`** | audio file | **MiniMax H3** | Same pipeline, segments rendered by MiniMax H3 instead of LTX. |
| **`music_video_MINIMAX_AUDIO_TO_LTX_VIDEO.json`** | MiniMax Music 3 | LTX 2.5 | Generating custom music *and* video from prompts, rendered by LTX. |
| **`music_video_MINIMAX_AUDIO_TO_MINIMAX_VIDEO.json`** | MiniMax Music 3 | **MiniMax H3** | End-to-end MiniMax: caption in, song and video out. |

### Modular Step Workflows (Parts 1, 2, 3)
| Workflow File | Backend | Description |
| :--- | :--- | :--- |
| **`part1_song_to_timeline.json`** | audio file | Step 1 &mdash; load song, audio analysis, lyrics, timeline, subject mode, card prompts. |
| **`part1_song_to_timeline_ltx.json`** | audio file | Step 1, LTX-named copy. |
| **`part1_song_to_timeline_minimax.json`** | MiniMax Music 3 | Step 1 &mdash; type caption & lyrics, generate the song, then analyse and plan it. |
| **`part2_storyboard_cards.json`** | image model | Step 2 &mdash; turnaround reference sheets and keyframe cards. |
| **`part2_storyboard_cards_ltx.json`** | image model | Step 2, LTX-named copy. |
| **`part2_storyboard_cards_minimax.json`** | image model | Step 2, MiniMax-named copy. |
| **`part3_cards_to_video.json`** | LTX 2.5 | Step 3 &mdash; card deck assembly, video segments, audio stitching. |
| **`part3_cards_to_video_ltx.json`** | LTX 2.5 | Step 3, LTX-named copy. |
| **`part3_cards_to_video_minimax.json`** | **MiniMax H3** | Step 3 rendered by MiniMax H3. |

> **Note on the two video backends.** Part 2 is backend-agnostic &mdash; it renders cards
> with the image model, so the `_ltx` and `_minimax` copies of Part 2 are the same graph
> and exist so each set is complete. Part 3 and the All-In-One files differ for real:
> the LTX files use `LTXSegmentsRender`, the MiniMax files use `MiniMaxSegmentsRender`,
> each wired to its own transformer, text encoder and VAEs.

| | LTX 2.5 | MiniMax H3 |
| :--- | :--- | :--- |
| Render node | `LTXSegmentsRender` | `MiniMaxSegmentsRender` |
| Transformer | `ltx-2.5-22b-distilled-transformer-*` | `minimax_h3_fl2va_*` |
| Text encoder | Gemma 3 12B (`type: ltxv`) | Qwen3-VL 32B (`type: minimax`) |
| VAEs | `ltx-2.5-video-vae`, `ltx-2.5-audio-vae` | `minimax_h3_video_vae`, `minimax_h3_audio_vae` |
| Guidance | Dual-CFG, positive + negative | Guidance-free &mdash; **no negative prompt** |
| Clip length | frames&nbsp;&minus;&nbsp;1 divisible by 8 (&plusmn;0.17s at 24 fps) | 5&nbsp;+&nbsp;17k frames (&plusmn;0.35s at 24 fps), trimmed on stitch |

Both render up to `segments_per_queue` segments per execution (default 4), save each clip
into the run as it finishes, and skip completed clips on the next queue. This keeps peak
memory bounded for long songs. Set ComfyUI's Queue batch count from the status panel to
run the remaining batches without manually pressing Queue each time.

### Standalone
| Workflow File | Description |
| :--- | :--- |
| **`audio_minimax_music_3.json`** | Text-to-Music with MiniMax Music 3 on its own &mdash; caption & lyrics in, song out. No storyboard or video. |

---

## Installation

### Option A &mdash; ComfyUI Manager
Search for **"Geekatplay VideoClipMaker"** and click Install.

### Option B &mdash; Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/GeekatplayStudio/ComfyUI-Music-to-Video.git
```
Then install the dependencies:

- **Windows portable ComfyUI** &mdash; double-click `install.bat` (finds the embedded Python).
- **Linux / macOS** &mdash; `bash install.sh` with your ComfyUI Python active.
- **Manually** &mdash; `pip install -r requirements.txt` in your ComfyUI environment.

Restart ComfyUI. The nodes appear under **Geekatplay/VideoClipMaker**.

### Requirements

| | |
| :--- | :--- |
| **ComfyUI** | v0.30+ (core `LoadAudio`, `ResolutionSelector`, `PreviewAny`, `PreviewImage`) |
| **ffmpeg + ffprobe** | **required** &mdash; on PATH. Joins clips, muxes your song, measures songs given by path. |
| **librosa** | installed by the installer. Drives tempo/key/energy analysis and beat detection. Without it the pipeline still runs, minus music-based styling and beat-cutting. |
| **Lyric transcription** | optional. Uses the first backend found: `faster-whisper`, `openai-whisper`, then **transformers** (already shipped with ComfyUI) &mdash; so it usually works with no extra install. With none, set `whisper_mode` to `none` and paste lyrics into `custom_lyrics`. |
| **Image model** | any. The workflows ship wired for **Z-Image Turbo** (UNet + Qwen3-4B text encoder + 16-channel VAE) at 8 steps / cfg 1.0. |
| **Video model** | **LTX 2.5** (distilled transformer, Gemma 3 text encoder, video + audio VAE) **or MiniMax H3** (`minimax_h3_fl2va_*`, Qwen3-VL text encoder, video + audio VAE). Pick the matching workflow; both render the song in resumable, memory-safe batches. |
| **LLM prompt writer** | optional. Set `prompt_writer` to **local LLM** and the card and motion prompts are written by a small instruct model that has read the song, instead of by the keyword tables. Needs `transformers` + `accelerate` (both ship with ComfyUI) and an instruct checkpoint in `ComfyUI/models/LLM` or named as a HF repo id. Falls back to the tables if it cannot load, so it is never required. |

---

## The workflows

| File | Use it for |
| :--- | :--- |
| **`music_video_ALL_IN_ONE.json`** | **Start here.** Everything in one graph: two stops, then resumable video batches. |
| `part1_song_to_timeline.json` | Plan only &mdash; song &rarr; lyrics &rarr; timeline &rarr; prompts &rarr; saved project. No GPU model loads, so it's fast to iterate on. |
| `part2_storyboard_cards.json` | Render references and cards for a saved project. |
| `part3_cards_to_video.json` | Render video segments and the final cut for a saved project. |

The three parts read and write the same run folder, so you can mix them: plan in Part 1,
render cards in Part 2, come back tomorrow and render video in Part 3.

---

## How it works

### 1. Song &rarr; timeline, cut on the beat

`AudioLyricsSegmenter` measures the song, transcribes the lyrics with timestamps, and
splits the track into segments of **5&ndash;10 seconds**, placing every word in the segment
it is sung in. It also analyses tempo, key, energy, brightness and percussiveness.

Segment boundaries are then **moved onto detected beats** wherever that keeps both
neighbours inside the 5&ndash;10s rule. Because each clip's length follows its segment
exactly, **every cut in the finished video lands on a beat of the song.** The video model
cannot hear your track &mdash; this is where the sense of rhythm actually comes from.

**N segments &rarr; N+1 cards.** A 3-minute song at 6s per segment gives 30 segments and 31
cards.

### 2. Stop 1 &mdash; review the plan

`SongTimelineReview` prints the whole mapping and holds the workflow there. You can:

- retime segments &mdash; type per-segment seconds into `segment_durations` (values are
  clamped to 5&ndash;10s and the timeline is rebuilt to span the song exactly)
- pick a **`style_preset`** &mdash; *Auto (match the music)* listens to the track and picks
  the look and palette from how it sounds, or force Photorealistic, Pencil drawing,
  Oil painting, Rococo, Abstract, Anime, Comic book, Film noir, Claymation, Action
  blockbuster&hellip;
- add a **`style_description`** in your own words &mdash; merged into every prompt

### 3. Prompts, derived from the song and music theme

`StoryboardPromptGenerator` builds every card prompt, every motion prompt, and the
reference plan dynamically driven by music analysis (tempo, energy, percussive beat grid, brightness, mode) and lyrics.

Before any prompt is written, the generator composes a **story**: one reading of
the song as a journey through several locations, with a light arc that travels
across the whole video (a minor-key song moves from deep night toward first
light; a major-key one burns from morning down to dusk) and a concrete action
per segment that advances the arc &mdash; opening, journey, rising, peak, turn,
resolution &mdash; mapped from each segment's detected musical section. Every card
is then a *different scene from that story*, the scene references are the
story's actual locations, and the review sheet opens with the full synopsis so
you can read the story before rendering a single image. The telling is seeded:
the same song and the same `llm_seed` retell the same story; change `llm_seed`
to get a different one.

> **Prompts are frozen into the run.** Parts 2 and 3 read the `prompt_list`
> stored in the run's `project.json`, written when Part 1 saved. After changing
> prompt settings (or updating this pack), re-run Part 1 so a new run carries
> the new prompts &mdash; re-queueing Part 2 alone renders the old ones.

**`prompt_writer`** chooses who actually writes them:

| Writer | What you get |
| :--- | :--- |
| **rule tables** *(default)* | Instant, offline, nothing to download. Imagery is matched from built-in keyword lists and the framing varies by card number. Reliable, but a song about a subject the tables never anticipated falls back to generic shots, and two songs in the same genre come out looking alike. |
| **local LLM** | A small instruct model reads the whole song **once** &mdash; lyrics, timeline and the measured audio analysis &mdash; decides who it is about, where it happens and what recurs, then writes every card and motion prompt against that one reading. Imagery comes from *this* song rather than from a list, and the deck stays coherent because one pass produced all of it. |

The LLM writer needs an instruct checkpoint: put a folder under `ComfyUI/models/LLM`
or type a Hugging Face repo id into `llm_model`. A local folder always wins, so an
offline install never reaches for the network.

**If you already run LTX 2.5 you do not need a new download.** LTX ships
**Gemma 3 12B Instruct**, which is a full instruct model with a chat template &mdash;
point `llm_model` at `Lightricks/gemma-3-12b-it-qat-q4_0-unquantized` and the writer
uses the weights already in your Hugging Face cache. Qwen-VL checkpoints kept for
captioning work the same way. Otherwise about 4B is plenty and downloads quickly
(`Qwen/Qwen3-4B-Instruct-2507` is the default). Anything larger than your card
holds is split across GPU and system RAM automatically rather than failing.

`llm_seed` makes a run reproducible &mdash; same song and seed give the same
storyboard, a new seed gives a different reading of the same song.

It runs in Part 1, before any renderer loads, and unloads itself when it is done.
Budget roughly a minute per eight segments. **If the model cannot be loaded, or
replies with something unusable, the node says so in the log and falls back to the
rule tables** &mdash; the fallback is whole-deck on purpose, because mixing model prose
and table prose in one storyboard reads worse than either on its own.

**`subject_mode`** selects how human figures or subjects appear in the video:

| Mode | Behavior & Reference Plan |
| :--- | :--- |
| **Auto** *(default)* | Smart auto-detection from lyrics & music theme &mdash; picks Singer, Band, Single Person, Crowd, or No People based on track analysis. |
| **Singer / Performer** | Focuses on a lead vocalist/performer with microphone, stage lighting sync, and expressive performance actions. |
| **Single Person / Main Character** | Focuses on a single protagonist whose narrative journey is visually tracked across cards. |
| **Band / Group** | Focuses on a multi-member band (vocalist, guitarist, bassist, drummer) performing in rhythm. |
| **Crowd / Atmospheric People** | Focuses on atmospheric crowds, dancers, or partygoers reacting to the music beat. |
| **No People (Pure Scenery)** | Eliminates human subjects &mdash; prompts focus on architecture, nature, environmental motion physics, and prop objects. **Character turnaround sheets are omitted**. |

Its `subject` / `setting` / `visual_style` / `motion_style` fields are
**optional overrides that ship empty** &mdash; left alone, the song decides.

**`prompt_approach`** chooses how the lyrics are read:

| Approach | What you get |
| :--- | :--- |
| **Creative** *(default)* | Reads the whole song first &mdash; dominant mood, recurring imagery &mdash; then each segment mixes its own imagery with an idea borrowed from another verse, plus a cinematic idea (colour shifts, particles, surreal touches, travelling light). Feels like a music video. |
| **Lyrics-focused** | Closely follows what is sung in that segment: its actions, objects, emotions, places. |
| **Abstract** | Paints the mood rather than the words &mdash; colour, form and motion, the subject as a presence rather than a portrait. |

Camera movement, dynamic lighting, and song structure dynamics are driven by **adaptive NLP & parametric audio signal analysis**:
- **Adaptive NLP Lyric Analysis**: Dynamically extracts action verbs, visual nouns, sensory descriptors, and emotional tone for ANY lyric line without relying on rigid static word tables.
- **Multi-Feature Signal Analysis**: Extracts Tempo (BPM), Beat Grid, Peak-Normalised Energy, Harmonic vs Percussive Ratio (HPSS), Spectral Centroid (Brightness), Spectral Flatness, Spectral Contrast (dB), Spectral Rolloff, and Zero Crossing Rate (ZCR).
- **Continuous Parametric Audio Feature Synthesis**: Dynamically calculates lighting contrast, specular rim intensity, color palette, camera velocity, and motion physics as continuous functions of audio metrics.
- **Emotion & Affect Mapping (Arousal & Valence)**: Maps audio features to emotional dimensions (*Triumphant/Euphoric*, *Intense/Aggressive*, *Peaceful/Intimate*, *Melancholic/Aching*, *Driving/Kinetic*).
- **Song Structure Annotation**: Analyzes time-varying energy curves to automatically identify structural song sections (*Intro*, *Verse*, *Build-up*, *Chorus / Climax*, *Bridge*, *Outro*) and adjust visual intensity accordingly.
  - **Chorus / Climax**: peak visual contrast, explosive rim lighting, expansive framing.
  - **Build-up**: rising volumetric light beams, accelerating particle motion.
  - **Intro / Outro**: gentle diffused light, wide establishing framing, resolving slow camera pull-back.

### 4. References &mdash; decided by the lyrics

Consistency is the hard part: the same person must be the same person in card 3 and card
30, and the guitar in the chorus must be the guitar from the first verse.

| Kind | Created when | Angles rendered |
| :--- | :--- | :--- |
| **character** | when a subject is present (`subject_mode` != No People) | front &middot; three-quarter &middot; profile &middot; back &middot; portrait &middot; wardrobe detail |
| **scene** | every location the lyrics keep returning to, plus the song's own setting | wide &middot; reverse &middot; high &middot; eye level &middot; detail &middot; empty |
| **prop** | an object named in **two or more segments** &mdash; a letter, a clock, a guitar, a car | front &middot; three-quarter &middot; side &middot; back &middot; macro &middot; in hands |

A song naming one place and no objects gets 2 references; one that travels and carries
objects gets as many as it needs. **A reference no card uses is dropped** rather than
rendered for nothing. One picture only pins down the angle it happens to show &mdash; a
turnaround describes the whole design, so the cards stop re-inventing what that one view
never showed.

### 5. Cards &mdash; all of them, in one queue

`StoryboardCardsRender` expands itself into one sampler chain per card at execution time,
each built on the reference assigned to it (close shots on the character, wide shots on
the location, tight shots on the prop). Ten cards means ten renders from one queue.

`reference_strength` controls how tightly: 0 ignores the reference, 1 hugs it. 0.5 is a
good start.

### 6. Stop 2 &mdash; review the deck

`StoryboardCardReview` checks the deck against the card count the timeline requires and
holds there until you flip `approve_cards`.

### 7. Video &mdash; resumable, memory-safe batches

`LTXSegmentsRender` expands into one complete LTX 2.5 first/last-frame chain **per
segment**. It renders `segments_per_queue` unfinished segments per execution (default 4),
saves each clip immediately, and skips saved clips next time. `MiniMaxSegmentsRender`
uses the same batching and resume behavior with the MiniMax H3 graph.

**Frame continuity:** consecutive segments *share a card*. Segment 1 renders
A1&rarr;A2, segment 2 renders **A2**&rarr;A3, segment 3 renders **A3**&rarr;A4. The same
image asset is reused &mdash; never regenerated &mdash; so every cut lands on a common frame.

While it runs you see, per segment: **NOW RENDERING** (which segment, its timecode, its
cards, its seed, the **full motion prompt sent to LTX**), the two frames being fed in, and
**CLIP SAVED** as each clip lands. Each clip is written the moment it finishes, so an
interrupted queue resumes for free.

**If LTX transitions cross-fade instead of moving:** lower `end_guide_strength` (default
`0.7`). It controls how hard each clip is pulled onto the next card &mdash; nearer 1.0
dissolves, lower gives real motion. The first frame is always pinned at 0.7 so every clip
starts exactly on its card.

### 8. Final cut

`VideoSegmentStitcher` joins **only this run's clips**, gives each segment its own slice
of the song, and muxes the full original soundtrack. Any segment not yet rendered is named
in the report rather than silently skipped.

---

## Where your work is saved

Everything a run produces stays inside that run's own folder &mdash; nothing is written to
the shared `output/video` folder, so clips from one project can never be mistaken for
another's.

```
output/storyboard_projects/<project>__<date-time>__<id>/
├── project.json            timeline, lyrics, style, prompts, reference plan
├── song.wav                a copy of your song, so the final mux always finds it
├── references/             every reference angle + references.txt
├── cards/                  card_0000.png ...
├── clips/                  clip_000.mp4 ... straight from LTX, one per segment
├── segments/               segment_000.mp4 ... each with its own slice of the song
└── final_music_video.mp4   the whole video with the full original soundtrack
```

The short `id` is what makes a run an identity: a timestamp alone is not enough, since two
queues in the same second or two projects sharing a name would collide. **Nothing is ever
overwritten** &mdash; delete the runs you don't want.

**`run_mode`** on the project-save node:

- **`new run every queue`** *(Part 1 default)* &mdash; each queue writes its own project.
- **`one run per song + timeline`** *(all-in-one default)* &mdash; the same song, timeline
  and style always resolve to the same folder, so every execution of one project writes
  into it. Change the song or the timing and you get a new run.

---

## Node reference

All nodes live under **`Geekatplay/VideoClipMaker`**, numbered in pipeline order.

| Node | What it does |
| :--- | :--- |
| **`MusicVideoStatus`** | Live dashboard: which step you're stopped on, progress for cards and clips, and the exact next action. Never cached. |
| **`AudioLyricsSegmenter`** | Song &rarr; duration, tempo/key/energy, beat grid, timed lyrics, beat-snapped segments, card count. |
| ⏸ **`SongTimelineReview`** | **Stop 1.** Timeline table, retiming, style preset + description. |
| **`StoryboardPromptGenerator`** | Every card prompt, every motion prompt, the reference plan, the review sheet. |
| **`StoryboardProjectSave`** | Writes the run folder, copies the song into it, stores the whole plan. |
| **`StoryboardProjectLoad`** | Opens a saved run for the standalone Part 2 / Part 3 workflows. |
| **`StoryboardReferencesRender`** | Renders every reference as a turnaround sheet in one queue. |
| **`StoryboardReferenceSaver`** | Writes the sheet to `<run>/references/` with `references.txt`. |
| **`StoryboardCardsRender`** | Renders **all** N+1 cards in one queue, each on its reference. |
| **`StoryboardCardSaver`** | Writes the deck into `<run>/cards/`. |
| ⏸ **`StoryboardCardReview`** | **Stop 2.** Validates the deck against the required count. |
| **`StoryboardCardLoader`** | Gathers a deck from disk (standalone Part 3). |
| **`StageMemoryCleaner`** | Unloads models from VRAM between stages. |
| **`LTXSegmentsRender`** | Renders LTX 2.5 segments in resumable batches and reports the live feed. |
| **`MiniMaxSegmentsRender`** | Renders MiniMax H3 segments with the same batching and resume behavior. |
| **`SegmentVideoSaver`** | Writes each clip to `<run>/clips/` under its segment number. |
| **`VideoSegmentStitcher`** | Joins the clips, slices the song per segment, muxes the final cut. |
| **`KeyframePairBatcher`** | *Utility.* Manual per-segment mode. |
| **`DynamicCardBatchPrompter`** | *Utility.* One card prompt per queue. |
| **`StoryboardCardSelector`** | *Utility.* Inspect one card's prompt, lyrics and timing. |

---

## Troubleshooting

| Symptom | Cause / fix |
| :--- | :--- |
| Nothing happens when you queue | You're at a stop. Check the **STOP** node's panel &mdash; it says what to flip. |
| Panels are blank on the canvas | Hard-refresh the browser (**Ctrl+F5**). The pack ships a JS extension that draws these panels and the browser caches it. |
| Node changes don't take effect | Restart ComfyUI. Python does not reload edited node modules. |
| `No saved runs yet` | Run Part 1 first, or use the all-in-one workflow. |
| Part 2's `run` dropdown doesn't list the run you just made | The list is built when the page loads. Press **`refresh run list`** on the Project Load node &mdash; it also refreshes itself whenever a queue finishes. `<< newest run >>` always resolves correctly regardless. |
| Final video is shorter than the song | Some segments haven't rendered. The stitcher names them in its report. |
| Video cross-fades instead of moving | Raise `end_guide_strength` toward 1.0 only if you want a dissolve; to get *more motion* change the prompt, not this value. |
| Segments **cut** instead of flowing into each other | `end_guide_strength` is too low. The official LTX 2.5 first/last-frame workflow pins **both** ends at **0.7**, and that is the default here. Below it, a clip drifts away from the card it should land on while the next clip starts exactly on that card &mdash; so every boundary jumps. Note a small residual remains regardless: the stitcher trims each clip to its segment length, which drops the last frame or two. |
| Prompts don't match the song | Leave `subject` / `setting` empty so they're derived; check the review sheet, which prints `derived from the lyrics` or `typed by you`. |
| `ffprobe was not found` | Install ffmpeg and put it on PATH, or feed the song through a `Load Audio` node instead of a path. |
| No music-based styling, 0 BPM | `librosa` is missing &mdash; `pip install librosa`. |
| `hostbuf_file_reader_read failed` / `HostBuffer.read_file_slice failed` / `The paging file is too small` | Windows **commit memory** (RAM + page file) ran out while dynamic VRAM streamed model weights through pinned host buffers. It does not fail as a clean OOM: the host read fails, CUDA then reports `out of memory`, and the **prompt worker thread dies** &mdash; the server stays up but can no longer run anything, and its VRAM stays allocated until you restart it. **Fix: launch with `--disable-dynamic-vram`** (use `run_nvidia_gpu_stable_memory.bat`), and/or raise the Windows page file. Check free commit with `Get-CimInstance Win32_OperatingSystem` &rarr; `FreeVirtualMemory`; if it is near zero, this is your error. |

Every workflow routes its data through a **Stage Memory Cleaner** at each stage
boundary (after transcription/music generation, before the card render, before
the video render). The cleaner unloads all models and empties the CUDA cache,
so whisper, the music model, the image model and the video model are never held
at the same time. It caches with its input on purpose: queueing Part 2 eighty
times for a card batch cleans once at the start, not between every card.

---

## Package contents

```
ComfyUI-Music-to-Video/
├── README.md                   this file
├── LICENSE                     MIT
├── nodes.py                    node implementations
├── __init__.py                 node registration
├── llm_writer.py               optional local-LLM prompt writer (see prompt_writer)
├── requirements.txt            Python dependencies
├── pyproject.toml              ComfyUI Registry / Manager metadata
├── install.bat                 Windows installer (portable-ComfyUI aware)
├── install.sh                  Linux / macOS installer
├── web/js/                      status, review, and segment-progress panels
└── example_workflows/
    ├── music_video_ALL_IN_ONE.json                  start here
    ├── music_video_MINIMAX_AUDIO_TO_LTX_VIDEO.json  same, song generated not loaded
    ├── part1_song_to_timeline.json
    ├── part1_song_to_timeline_minimax.json          Part 1 with a generated song
    ├── part2_storyboard_cards.json
    ├── part3_cards_to_video.json
    └── audio_minimax_music_3.json                   text-to-music on its own
```

---

<div align="center">

### Geekatplay Studio &mdash; Vladimir Chopine

**Thank you for your support!**

[GitHub](https://github.com/GeekatplayStudio/ComfyUI-Music-to-Video) &middot;
[@geekatplay](https://www.youtube.com/@geekatplay) &middot;
[@geekatplay-ru](https://www.youtube.com/@geekatplay-ru) &middot;
[@v-code-studio](https://www.youtube.com/@v-code-studio)

*Released under the MIT License.*

</div>
