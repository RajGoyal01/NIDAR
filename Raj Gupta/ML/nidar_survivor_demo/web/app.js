"use strict";
const $ = (id) => document.getElementById(id);
let latest = null,
  lastSuccess = 0,
  frameUrl = null,
  frameBusy = false;
let frameSequence = null, frameTimes = [];
window.nidarPerformance = { displayedFrames: 0, uniqueDecodedFps: 0 };
const show = (value) =>
  value === null || value === undefined ? "—" : String(value);
const node = (tag, text, className) => {
  const el = document.createElement(tag);
  el.textContent = text;
  if (className) el.className = className;
  return el;
};
const timeLabel = (value) => {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? "—"
    : date.toLocaleTimeString([], { hour12: false });
};
const titles = {
  survivor_created: "New identity accepted",
  duplicate_prevented: "Returning identity • duplicate avoided",
  identity_pending: "Identity awaiting evidence",
  survivor_not_visible: "Identity left view",
  perception_unavailable: "Perception unavailable",
  mission_started: "Mission started",
  mission_resumed: "Mission restored",
  mission_closed: "Mission saved",
  replay_complete: "Offline replay complete",
};
function hideFrame(title, message) {
  $("video").hidden = true;
  $("empty").hidden = false;
  $("empty-title").textContent = title;
  $("empty-message").textContent = message;
}
function renderRows() {
  if (!latest) return;
  const query = $("search").value.trim().toLowerCase();
  const rows = latest.records.filter((r) =>
    `${r.survivor_id} ${r.reference_id}`.toLowerCase().includes(query),
  );
  $("record-total").textContent = latest.records.length;
  $("records").replaceChildren();
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.append(node("td", r.survivor_id), node("td", r.reference_id));
    const status = document.createElement("td");
    status.append(
      node(
        "span",
        r.status.replaceAll("_", " "),
        `tag ${r.status === "VISIBLE" ? "visible" : ""}`,
      ),
    );
    tr.append(
      status,
      node("td", timeLabel(r.last_seen_at)),
      node(
        "td",
        typeof r.identity_score === "number"
          ? r.identity_score.toFixed(3)
          : "—",
      ),
    );
    $("records").append(tr);
  }
  if (!rows.length) {
    const tr = document.createElement("tr"),
      td = node(
        "td",
        query
          ? "No identities match this search."
          : "No accepted identities yet. Uncertain observations stay pending.",
        "empty-cell",
      );
    td.colSpan = 5;
    tr.append(td);
    $("records").append(tr);
  }
}
function renderEvents() {
  if (!latest) return;
  const filter = $("filter").value;
  const events = latest.events
    .filter(
      (e) =>
        filter === "all" ||
        (filter === "identity") === /survivor|duplicate|identity/.test(e.event),
    )
    .slice()
    .reverse();
  $("events").replaceChildren();
  for (const e of events) {
    const li = document.createElement("li"),
      body = node("div", "", "event-text");
    body.append(
      node("strong", titles[e.event] || e.event.replaceAll("_", " ")),
      node(
        "p",
        [e.survivor_id, e.reference_id, e.track_id, e.reason]
          .filter(Boolean)
          .join(" · "),
      ),
    );
    li.append(
      node("span", "↳", "event-icon"),
      body,
      node("time", timeLabel(e.recorded_at)),
    );
    $("events").append(li);
  }
  if (!events.length) $("events").append(node("li", "No events in this view."));
}
function render(data) {
  if (latest?.mission_id !== data.mission_id) {
    frameSequence = null;
    frameTimes = [];
    hideFrame("New mission", "Waiting for fresh evidence.");
  }
  latest = data;
  $("reset-demo").disabled = !data.reset?.available || data.reset.pending;
  $("reset-demo").textContent = data.reset?.pending ? "Saving and resetting…" : "New Mission / Reset Demo";
  $("previous-missions").replaceChildren();
  for (const old of data.previous_missions || []) {
    const link = node("a", `${old.mission_id.slice(0, 8)} · ${old.unique_survivors} identities · Export saved report`);
    link.href = `/api/archive/${encodeURIComponent(old.mission_id)}`;
    link.download = "previous-mission.json";
    $("previous-missions").append(link, document.createElement("br"));
  }
  $("mode").textContent = data.mode;
  $("mission").textContent = data.mission_id
    ? `MISSION ${data.mission_id.slice(0, 8).toUpperCase()}`
    : "No mission loaded";
  $("connection").textContent = data.status;
  $("connection").className =
    `pill ${data.status === "ONLINE" ? "online" : ""}`;
  $("current").textContent = show(data.counts.current_persons);
  $("unique").textContent = show(data.counts.unique_survivors);
  $("duplicates").textContent = show(data.counts.duplicates_prevented);
  $("pending").textContent = show(data.counts.pending_persons);
  $("notice").textContent =
    data.mode === "ARCHIVE"
      ? "SAVED MISSION · Historical records only. Current occupancy and live module health are unknown. No camera recording was stored."
      : data.mode === "OFFLINE REPLAY"
        ? "OFFLINE LABELLED REPLAY · Real appearance model, known tracks and synthetic view changes. Not live-camera accuracy."
        : "LIVE PERCEPTION · Unique identities are appearance-based estimates, not medically verified survivors.";
  $("feed-label").textContent =
    data.mode === "ARCHIVE" ? "NO RECORDING" : data.mode;
  if (data.mode === "LOCAL VIDEO")
    $("notice").textContent =
      "LOCAL VIDEO · Full perception pipeline on a file, not a live phone camera. Unique identities remain estimates.";
  $("stage").textContent = data.stage
    ? data.stage.replaceAll("_", " ").toUpperCase()
    : data.status;
  $("modules").replaceChildren();
  for (const [name, status] of Object.entries(data.modules)) {
    const row = node("div", "", "module");
    row.append(
      node("span", name),
      node("span", status, status === "ACTIVE" ? "good" : ""),
    );
    $("modules").append(row);
  }
  const metric = (id, key, unit) => {
    const v = data.metrics[key];
    $(id).textContent = typeof v === "number" ? `${v.toFixed(1)} ${unit}` : "—";
  };
  metric("capture", "capture_fps", "fps");
  metric("processed", "processed_fps", "fps");
  metric("inference", "inference_ms", "ms");
  metric("age", "decoded_age_ms", "ms");
  if (!data.frame_available)
    hideFrame(
      data.mode === "ARCHIVE"
        ? "Mission saved. No video recorded."
        : data.status === "COMPLETE"
          ? "Replay complete"
          : "Awaiting a fresh frame",
      data.status === "COMPLETE"
        ? "Review the accepted identities and decision trail below."
        : "Current occupancy is unknown when fresh perception is unavailable.",
    );
  $("updated").textContent =
    `STATUS RECEIVED ${new Date().toLocaleTimeString([], { hour12: false })}`;
  renderRows();
  renderEvents();
}
function disconnected() {
  $("reset-demo").disabled = true;
  $("connection").textContent = "DISCONNECTED";
  $("connection").className = "pill";
  $("notice").textContent =
    "BACKEND UNREACHABLE · Historical totals below are last-known, not current observations. Reconnecting automatically…";
  for (const id of [
    "current",
    "pending",
    "capture",
    "processed",
    "inference",
    "age",
  ])
    $(id).textContent = "—";
  hideFrame(
    "Connection interrupted",
    "No stale video is shown as live. Keep the local backend running.",
  );
  if (latest) {
    latest = {
      ...latest,
      records: latest.records.map((r) => ({ ...r, status: "UNKNOWN" })),
    };
    renderRows();
  }
  $("modules")
    .querySelectorAll(".module span:last-child")
    .forEach((el) => {
      el.textContent = "UNKNOWN";
      el.className = "";
    });
}
async function poll() {
  try {
    const r = await fetch("/api/state", {
      cache: "no-store",
      signal: AbortSignal.timeout(2000),
    });
    if (!r.ok) throw Error("status");
    render(await r.json());
    lastSuccess = Date.now();
  } catch {
    disconnected();
  } finally {
    setTimeout(poll, 350);
  }
}
async function frame() {
  const started = performance.now();
  if (
    !frameBusy &&
    latest?.frame_available &&
    Date.now() - lastSuccess < 1500
  ) {
    frameBusy = true;
    try {
      const r = await fetch("/api/frame", {
        cache: "no-store",
        signal: AbortSignal.timeout(1500),
      });
      if (r.status === 200) {
        const frameMission = r.headers.get("X-Mission-ID");
        const sequence = r.headers.get("X-Frame-Sequence");
        const blob = await r.blob();
        if (sequence !== frameSequence && Date.now() - lastSuccess < 1500 && latest?.frame_available) {
          const url = URL.createObjectURL(blob);
          const decoded = new Image();
          decoded.src = url;
          try { await decoded.decode(); } catch (error) { URL.revokeObjectURL(url); throw error; }
          if (Date.now() - lastSuccess >= 1500 || !latest?.frame_available || frameMission !== latest.mission_id) {
            URL.revokeObjectURL(url);
            throw Error("Frame expired during decode");
          }
          $("video").src = url;
          $("video").hidden = false;
          $("empty").hidden = true;
          if (frameUrl) URL.revokeObjectURL(frameUrl);
          frameUrl = url;
          frameSequence = sequence;
          const now = performance.now();
          frameTimes.push(now);
          frameTimes = frameTimes.filter(t => now - t <= 2000);
          window.nidarPerformance.displayedFrames++;
          window.nidarPerformance.uniqueDecodedFps = frameTimes.length > 1
            ? (frameTimes.length - 1) * 1000 / (now - frameTimes[0]) : 0;
        }
      } else hideFrame("Awaiting a fresh frame", "Stale frames are withheld.");
    } catch {
      hideFrame("Frame unavailable", "Retrying the latest frame.");
    } finally {
      frameBusy = false;
    }
  }
  setTimeout(frame, Math.max(1, 1000 / (latest?.preview_fps_limit || 30) - (performance.now() - started)));
}
$("search").addEventListener("input", renderRows);
let resetMission = null;
$("reset-demo").addEventListener("click", () => {
  resetMission = latest?.mission_id;
  $("reset-dialog").showModal();
});
$("reset-cancel").addEventListener("click", () => $("reset-dialog").close());
$("reset-confirm").addEventListener("click", async () => {
  $("reset-confirm").disabled = true;
  try {
    const response = await fetch("/api/mission/reset", {
      method: "POST", headers: {"X-Reset-Token": latest?.reset?.token || "", "X-Mission-ID": resetMission || ""},
      signal: AbortSignal.timeout(5000)
    });
    if (!response.ok) throw Error("Mission changed or reset unavailable. Refresh status and try again.");
    $("reset-dialog").close();
    $("reset-demo").disabled = true;
    $("notice").textContent = "Reset requested. Waiting for the backend to save and switch missions.";
  } catch (error) {
    $("reset-dialog").close();
    $("notice").textContent = `${error.message} Check the mission ID before retrying.`;
  } finally { $("reset-confirm").disabled = false; }
});
$("filter").addEventListener("change", renderEvents);
$("fullscreen").addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $("feed").requestFullscreen();
  } catch {
    $("empty-message").textContent =
      "Fullscreen is unavailable in this browser.";
  }
});
setInterval(() => {
  if (lastSuccess && Date.now() - lastSuccess > 2000) disconnected();
}, 500);
poll();
frame();
