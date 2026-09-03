/*
 * ComfyUI-Music-to-Video  -  Geekatplay Studio, Vladimir Chopine
 *
 * Step 3 progress board.
 *
 * The console prints one banner per segment and it scrolls away, so while a
 * song renders you cannot see how many segments there are, which are already
 * on disk, or which pair of cards is being animated right now. This draws the
 * WHOLE timeline as rows: the two cards LTX is interpolating between, the
 * prompt driving them, and the finished clip appearing in the same row the
 * moment it is written.
 *
 * Fed by "gap_board" payloads: the render node sends the full plan, each
 * segment banner sends "now", each saved clip sends "done". Expanded-subgraph
 * ui payloads land on the parent render node, which is where this listens.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const BOARD_NODES = { LTXSegmentsRender: 1, MiniMaxSegmentsRender: 1, SegmentTable: 1 };

const STATUS_STYLE = {
  done:    { label: "done",      color: "#7fd67f", border: "#2f6b34" },
  now:     { label: "rendering", color: "#ffb44d", border: "#8a5a12" },
  queued:  { label: "queued",    color: "#8fb6ff", border: "#2b4a80" },
  pending: { label: "later",     color: "#7c8798", border: "#2b3340" },
};

function viewURL(ref) {
  if (!ref || !ref.filename) return null;
  const query = new URLSearchParams({
    filename: ref.filename,
    subfolder: ref.subfolder || "",
    type: ref.type || "output",
  });
  return api.apiURL(`/view?${query}`);
}

function boardState(node) {
  if (!node.__gapBoard) node.__gapBoard = { rows: [], meta: {} };
  return node.__gapBoard;
}

function boardElement(node) {
  if (node.__gapBoardEl) return node.__gapBoardEl;

  const el = document.createElement("div");
  Object.assign(el.style, {
    overflow: "auto", width: "100%", height: "100%", boxSizing: "border-box",
    background: "#0e1116", border: "1px solid #2b3340", borderRadius: "4px",
    padding: "6px", fontFamily: "ui-monospace, Consolas, monospace",
    fontSize: "10px", color: "#c8d0da",
  });

  const widget = node.addDOMWidget("gap_board", "gap_board", el, {});
  // Node.serialize() skips a widget only on the DIRECT `serialize` property;
  // options.serialize alone is read by nothing and the widget would then claim
  // a widgets_values slot and shift every real widget on reload.
  widget.serialize = false;
  widget.serializeValue = () => undefined;
  widget.options = widget.options || {};
  widget.options.serialize = false;

  widget.computeSize = function (width) {
    const used = (node.widgets || [])
      .filter((w) => w !== widget)
      .reduce((h, w) => h + (w.computedHeight || LiteGraph.NODE_WIDGET_HEIGHT + 4), 0);
    const avail = (node.size?.[1] || 0) - used - LiteGraph.NODE_TITLE_HEIGHT - 12;
    return [width, Math.max(220, avail)];
  };

  node.__gapBoardEl = el;
  return el;
}

function thumb(row, which) {
  const box = document.createElement("div");
  const index = which === "first" ? row.cardA : row.cardB;
  const url = viewURL(row[which]);
  if (url) {
    const img = document.createElement("img");
    img.src = url;
    img.loading = "lazy";
    img.title = `card ${index}`;
    Object.assign(img.style, { width: "72px", borderRadius: "2px", display: "block" });
    box.appendChild(img);
  } else {
    const empty = document.createElement("div");
    Object.assign(empty.style, {
      width: "72px", height: "40px", background: "#181d25",
      border: "1px dashed #2b3340", borderRadius: "2px",
    });
    empty.title = "card not rendered yet";
    box.appendChild(empty);
  }
  const caption = document.createElement("div");
  caption.style.color = "#6b7686";
  caption.textContent = `card ${index}${which === "first" ? " (first)" : " (last)"}`;
  box.appendChild(caption);
  return box;
}

function drawBoard(node) {
  const state = boardState(node);
  const el = boardElement(node);
  const rows = state.rows;

  if (!rows.length) {
    el.replaceChildren();
    const hint = document.createElement("div");
    hint.style.color = "#7c8798";
    hint.textContent = "Queue step 3 - every segment of the song will be listed here.";
    el.appendChild(hint);
    return;
  }

  const done = rows.filter((r) => r.status === "done").length;
  const header = document.createElement("div");
  Object.assign(header.style, {
    position: "sticky", top: "0", zIndex: "2", background: "#0e1116",
    paddingBottom: "4px", marginBottom: "6px",
    borderBottom: "1px solid #2b3340", color: "#e0d080",
  });
  header.textContent =
    `${state.meta.engine || "render"}   ${done}/${rows.length} segments rendered` +
    (state.meta.run ? `   run ${state.meta.run}` : "");
  el.replaceChildren(header);

  for (const row of rows) {
    const style = STATUS_STYLE[row.status] || STATUS_STYLE.pending;

    const line = document.createElement("div");
    Object.assign(line.style, {
      display: "grid",
      gridTemplateColumns: "58px 78px 78px 1fr 128px",
      gap: "6px", alignItems: "start", padding: "5px",
      marginBottom: "4px", borderRadius: "3px",
      border: `1px solid ${style.border}`,
      background: row.status === "now" ? "#1a1710" : "#12161c",
    });

    const label = document.createElement("div");
    label.style.color = style.color;
    label.style.whiteSpace = "pre";
    label.textContent = `seg ${row.i + 1}\n${style.label}`;
    line.appendChild(label);

    line.appendChild(thumb(row, "first"));
    line.appendChild(thumb(row, "last"));

    const text = document.createElement("div");
    text.style.lineHeight = "1.35";
    const when = document.createElement("div");
    when.style.color = "#8fb6ff";
    when.textContent = `${row.time}    ${row.dur}s -> ${row.frames} frames`;
    const prompt = document.createElement("div");
    prompt.style.color = "#c8d0da";
    prompt.style.whiteSpace = "pre-wrap";
    prompt.textContent = row.prompt || "(no prompt)";
    text.append(when, prompt);
    if (row.lyric && row.lyric !== "(instrumental)") {
      const lyric = document.createElement("div");
      lyric.style.color = "#7c8798";
      lyric.textContent = `lyric: ${row.lyric}`;
      text.appendChild(lyric);
    }
    line.appendChild(text);

    const clipBox = document.createElement("div");
    const clipURL = viewURL(row.clip);
    if (clipURL) {
      const video = document.createElement("video");
      video.src = clipURL;
      video.controls = true;
      video.loop = true;
      video.muted = true;
      video.preload = "metadata";
      Object.assign(video.style, { width: "128px", borderRadius: "2px", display: "block" });
      clipBox.appendChild(video);
    } else {
      Object.assign(clipBox.style, {
        width: "128px", height: "72px", background: "#181d25",
        border: "1px dashed #2b3340", borderRadius: "2px", color: "#6b7686",
        display: "flex", alignItems: "center", justifyContent: "center",
      });
      clipBox.textContent = row.status === "now" ? "animating..." : "not rendered";
    }
    line.appendChild(clipBox);

    el.appendChild(line);
    if (row.status === "now") {
      // Keep the segment being rendered in view without yanking the page.
      requestAnimationFrame(() => line.scrollIntoView({ block: "nearest" }));
    }
  }
}

function applyBoard(node, raw) {
  let message;
  try {
    message = JSON.parse(Array.isArray(raw) ? raw.join("") : raw);
  } catch (err) {
    return;
  }

  const state = boardState(node);
  if (message.kind === "plan") {
    state.rows = message.rows || [];
    state.meta = message;
  } else if (message.kind === "now") {
    // A row left mid-render by an interrupted queue must not stay "rendering".
    for (const row of state.rows) if (row.status === "now") row.status = "queued";
    const row = state.rows.find((r) => r.i === message.i);
    if (row) row.status = "now";
  } else if (message.kind === "done") {
    const row = state.rows.find((r) => r.segment_index === message.i)
             || state.rows.find((r) => r.i === message.i);
    if (row) {
      row.status = "done";
      if (message.clip) row.clip = message.clip;
    }
  } else {
    return;
  }
  drawBoard(node);
}

app.registerExtension({
  name: "Geekatplay.VideoClipMaker.SegmentBoard",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!BOARD_NODES[nodeData.name]) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const created = onCreated?.apply(this, arguments);
      boardElement(this);
      drawBoard(this);
      if ((this.size?.[1] || 0) < 460) this.size[1] = 460;
      return created;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = onExecuted?.apply(this, arguments);
      if (message?.gap_board) applyBoard(this, message.gap_board);
      return result;
    };
  },

  onNodeOutputsUpdated(outputs) {
    for (const [id, message] of Object.entries(outputs || {})) {
      if (!message?.gap_board) continue;
      const node = app.graph?.getNodeById?.(Number(id)) ?? app.graph?.getNodeById?.(id);
      if (node && BOARD_NODES[node.comfyClass]) applyBoard(node, message.gap_board);
    }
  },
});
