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

from .nodes import (
    AudioLyricsSegmenter,
    SongTimelineReview,
    StoryboardPromptGenerator,
    StoryboardCardSelector,
    DynamicCardBatchPrompter,
    StoryboardReferencesRender,
    StoryboardReferenceSaver,
    StoryboardCardsRender,
    StoryboardCardSaver,
    StoryboardCardLoader,
    StoryboardCardReview,
    KeyframePairBatcher,
    LTXSegmentsRender,
    SegmentTable,
    MiniMaxSegmentsRender,
    SegmentNowRendering,
    GAPLastFrame,
    SegmentVideoSaver,
    VideoSegmentStitcher,
    StageMemoryCleaner,
    MusicVideoStatus,
    StoryboardProjectSave,
    StoryboardProjectLoad,
)

NODE_CLASS_MAPPINGS = {
    "AudioLyricsSegmenter": AudioLyricsSegmenter,
    "SongTimelineReview": SongTimelineReview,
    "StoryboardPromptGenerator": StoryboardPromptGenerator,
    "StoryboardCardSelector": StoryboardCardSelector,
    "DynamicCardBatchPrompter": DynamicCardBatchPrompter,
    "StoryboardReferencesRender": StoryboardReferencesRender,
    "StoryboardReferenceSaver": StoryboardReferenceSaver,
    "StoryboardCardsRender": StoryboardCardsRender,
    "StoryboardCardSaver": StoryboardCardSaver,
    "StoryboardCardLoader": StoryboardCardLoader,
    "StoryboardCardReview": StoryboardCardReview,
    "KeyframePairBatcher": KeyframePairBatcher,
    "LTXSegmentsRender": LTXSegmentsRender,
    "SegmentTable": SegmentTable,
    "MiniMaxSegmentsRender": MiniMaxSegmentsRender,
    "SegmentNowRendering": SegmentNowRendering,
    "GAPLastFrame": GAPLastFrame,
    "SegmentVideoSaver": SegmentVideoSaver,
    "VideoSegmentStitcher": VideoSegmentStitcher,
    "StageMemoryCleaner": StageMemoryCleaner,
    "MusicVideoStatus": MusicVideoStatus,
    "StoryboardProjectSave": StoryboardProjectSave,
    "StoryboardProjectLoad": StoryboardProjectLoad,
}

# Numbered by the part of the pipeline they belong to, so the menu reads in the
# order the three workflows run.
NODE_DISPLAY_NAME_MAPPINGS = {
    # any part
    "MusicVideoStatus": "0. STATUS - where am I?",
    "StoryboardProjectLoad": "0b. LOAD project (Parts 2 and 3)",
    # part 1 - plan
    "AudioLyricsSegmenter": "1. Analyse song - lyrics and timeline",
    "SongTimelineReview": "1b. Timeline review and style (PAUSE)",
    "StoryboardPromptGenerator": "1c. Build card and motion prompts",
    "StoryboardProjectSave": "1d. SAVE project (hand off to Part 2)",
    # part 2 - references, then cards
    "StoryboardReferencesRender": "2. RENDER REFERENCES (turnaround sheets)",
    "StoryboardReferenceSaver": "2b. SAVE reference sheet (into the run)",
    "StoryboardCardsRender": "2c. RENDER ALL CARDS (one queue)",
    "StoryboardCardSaver": "2d. SAVE cards (into the run)",
    "StoryboardCardReview": "2e. Card review (PAUSE)",
    # part 3 - video
    "StoryboardCardLoader": "3. Assemble the card deck",
    "StageMemoryCleaner": "3b. Unload models (free VRAM)",
    "LTXSegmentsRender": "3c. RENDER ALL SEGMENTS (LTX 2.5, one queue)",
    "SegmentTable": "3d. SEGMENT TABLE (all segments, frames, clips)",
    "MiniMaxSegmentsRender": "3c. RENDER ALL SEGMENTS (MiniMax H3, one queue)",
    "KeyframePairBatcher": "util. Card pair monitor (manual per-segment mode)",
    "SegmentNowRendering": "util. Now-rendering banner (used by 3c)",
    "GAPLastFrame": "util. Last frame of a batch (used by 3c)",
    "SegmentVideoSaver": "3d. SAVE this segment's clip (into the run)",
    "VideoSegmentStitcher": "4. Join clips + original song = final cut",
    # utilities
    "DynamicCardBatchPrompter": "util. One card prompt per queue",
    "StoryboardCardSelector": "util. Inspect one card",
}

# Serves web/js/videoclipmaker.js, which renders the status/review text on the
# canvas. Without it ComfyUI only shows ui.text for its own PreviewAny node.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
