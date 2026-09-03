/*
 * ComfyUI-Music-to-Video  -  Geekatplay Studio, Vladimir Chopine
 * https://github.com/GeekatplayStudio/ComfyUI-Music-to-Video
 *
 * Thank you for your support! Star the project on GitHub and subscribe:
 *   https://www.youtube.com/@geekatplay   (English)
 *   https://www.youtube.com/@geekatplay-ru   (Russian)
 *   https://www.youtube.com/@v-code-studio   (V-Code Studio)
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/*
 * ComfyUI only renders a node's `ui.text` payload for its own PreviewAny node
 * (see the Comfy.PreviewAny extension: `node.type === "PreviewAny"`). Custom
 * nodes that return {"ui": {"text": [...]}} therefore display nothing at all -
 * their output only reaches the server console. This extension gives our
 * status / review nodes a readout panel so the pipeline state is visible on
 * the canvas, which is the whole point of those nodes.
 */

const READOUT_NODES = {
  MusicVideoStatus:     { min: 260, accent: "#ffb44d" },
  SongTimelineReview:   { min: 300, accent: "#7fd67f" },
  StoryboardCardReview: { min: 200, accent: "#7fd67f" },
  StoryboardCardSaver:  { min: 120, accent: "#8fb6ff" },
  StoryboardCardLoader: { min: 160, accent: "#8fb6ff" },
  StoryboardProjectSave: { min: 110, accent: "#8fb6ff" },
  StoryboardProjectLoad: { min: 260, accent: "#8fb6ff" },
  StoryboardPromptGenerator: { min: 300, accent: "#e0d080" },
  // Part 3 - what is actually being handed to LTX 2.5 for this segment.
  KeyframePairBatcher:  { min: 380, accent: "#ffb44d" },
  // Expanded banner/saver ui.text lands on this parent node - the live monitor.
  LTXSegmentsRender:    { min: 340, accent: "#ffb44d" },
  SegmentVideoSaver:    { min: 160, accent: "#7fd67f" },
  VideoSegmentStitcher: { min: 260, accent: "#8fb6ff" },
};

const PLACEHOLDER = "Queue the workflow to see this step's status.";

function readout(node) {
  if (node.__gapReadout) return node.__gapReadout;

  const cfg = READOUT_NODES[node.comfyClass] || { min: 160, accent: "#ccc" };
  const el = document.createElement("pre");
  Object.assign(el.style, {
    margin: "0",
    padding: "6px 8px",
    overflow: "auto",
    fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace",
    fontSize: "10px",
    lineHeight: "1.3",
    whiteSpace: "pre",
    color: cfg.accent,
    background: "#0e1116",
    border: "1px solid #2b3340",
    borderRadius: "4px",
    boxSizing: "border-box",
    width: "100%",
    height: "100%",
  });
  el.textContent = PLACEHOLDER;

  const widget = node.addDOMWidget("gap_readout", "gap_readout", el, {});

  // A widget that serializes would occupy a widgets_values slot and shift every
  // real widget when the workflow is reloaded. Node.serialize() skips a widget
  // ONLY on the direct `serialize` property - `options.serialize` alone is read
  // by nothing (ComfyUI's own audio widget sets both, in that order). Setting
  // just the option here silently corrupted saved workflows.
  widget.serialize = false;
  widget.serializeValue = () => undefined;   // keeps it out of the API prompt too
  widget.options = widget.options || {};
  widget.options.serialize = false;

  widget.computeSize = function (width) {
    const used = (node.widgets || [])
      .filter((w) => w !== widget)
      .reduce((h, w) => h + (w.computedHeight || LiteGraph.NODE_WIDGET_HEIGHT + 4), 0);
    const avail = (node.size?.[1] || 0) - used - LiteGraph.NODE_TITLE_HEIGHT - 12;
    return [width, Math.max(cfg.min, avail)];
  };

  node.__gapReadout = el;
  return el;
}

function paint(node, message) {
  if (!node || !READOUT_NODES[node.comfyClass]) return;
  let text = message?.text;
  if (Array.isArray(text)) text = text.join("");
  if (typeof text !== "string" || !text.length) return;
  readout(node).textContent = text;
}

app.registerExtension({
  name: "Geekatplay.VideoClipMaker.Readout",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!READOUT_NODES[nodeData.name]) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      readout(this);
      return r;
    };

    // Classic per-node hook - still dispatched for OUTPUT_NODE ui payloads.
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      paint(this, message);
      return r;
    };
  },

  // Newer frontend path, and it also fires when a finished run is restored.
  onNodeOutputsUpdated(outputs) {
    for (const [id, message] of Object.entries(outputs || {})) {
      const node = app.graph?.getNodeById?.(Number(id)) ?? app.graph?.getNodeById?.(id);
      paint(node, message);
    }
  },
});

/*
 * Run list refresh.
 *
 * StoryboardProjectLoad's `run` dropdown is built by INPUT_TYPES, which the
 * server evaluates once and the browser caches when the page loads. Approving
 * Part 1 writes a NEW run folder, but Part 2's dropdown still shows the list
 * from page load - so "<< newest run >>" is right while every named run is
 * stale, and a run you just made cannot be picked without reloading the tab.
 *
 * Re-fetching this one node's definition asks the server to run INPUT_TYPES
 * again, which re-lists the run folders.
 */
const RUN_LIST_NODE = "StoryboardProjectLoad";
const RUN_WIDGET = "run";

async function refreshRunList(quiet) {
  let values;
  try {
    const resp = await api.fetchApi(`/object_info/${RUN_LIST_NODE}`, { cache: "no-store" });
    const def = (await resp.json())?.[RUN_LIST_NODE];
    values = def?.input?.required?.[RUN_WIDGET]?.[0];
  } catch (err) {
    console.error("[VideoClipMaker] could not refresh the run list:", err);
    return 0;
  }
  if (!Array.isArray(values) || !values.length) return 0;

  let touched = 0;
  for (const node of app.graph?.nodes || []) {
    if (node.comfyClass !== RUN_LIST_NODE) continue;
    const widget = (node.widgets || []).find((w) => w.name === RUN_WIDGET);
    if (!widget) continue;

    const previous = widget.value;
    widget.options = widget.options || {};
    widget.options.values = [...values];
    // Keep the chosen run selected; if that folder is gone, fall back to the
    // newest rather than leaving a value the server would reject.
    if (!values.includes(previous)) widget.value = values[0];
    touched++;
    if (!quiet && previous !== widget.value) {
      console.log(`[VideoClipMaker] run '${previous}' is gone - selected '${widget.value}'.`);
    }
  }
  if (touched) app.graph?.setDirtyCanvas?.(true, true);
  return touched;
}

app.registerExtension({
  name: "Geekatplay.VideoClipMaker.RunListRefresh",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== RUN_LIST_NODE) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const button = this.addWidget("button", "refresh run list", null, async () => {
        button.label = "refreshing...";
        app.graph?.setDirtyCanvas?.(true, true);
        const n = await refreshRunList(false);
        button.label = n ? "refresh run list" : "refresh failed - see console";
        app.graph?.setDirtyCanvas?.(true, true);
      });
      // Must be the direct property: see the readout widget above. A button
      // carries no value worth saving, and a serialising one would shift `run`.
      button.serialize = false;
      button.serializeValue = () => undefined;
      button.options = button.options || {};
      button.options.serialize = false;
      return r;
    };
  },

  setup() {
    // Part 1 finishing is exactly when a new run appears, so pick it up without
    // making anyone press the button. Debounced: a queue of 80 cards fires this
    // once per card.
    let pending = null;
    const schedule = () => {
      clearTimeout(pending);
      pending = setTimeout(() => refreshRunList(true), 400);
    };
    for (const event of ["execution_success", "executed"]) {
      api.addEventListener(event, schedule);
    }
  },
});
