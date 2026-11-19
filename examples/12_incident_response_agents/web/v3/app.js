// The approvals page, v3 (v2's script, a quieter stylesheet): a client of the relay's /incidents API (relay.py). Every value from the API is shown as text
// (textContent), never as HTML. The operator token stays in this tab's session storage and goes in the Authorization header.
//
// Two views. Incident: a rail of incidents, and the selected one as a story (steps, the commander's latest words, the decision,
// its agent hosts, one MicroVM each, then Story / Activity / Action Gateway / Postmortem tabs). Fleet: one lane per incident,
// every agent host side by side. What is read when:
//   - every 3 s the relay's own list (it wakes no host): status, pending, latest run, alert, agent hosts, latest tool call;
//   - an incident (its host) only when the list shows it changed, when it is selected, or while one of its runs works,
//     and never while its commander's host is still starting (or failed to start): there is no host to read yet;
// so a commander waiting for a decision can pause in its host.
"use strict";

const $ = (id) => document.getElementById(id);
const ROLES = ["metrics", "changes", "infra", "remediator", "verifier"];
// What each agent is called on the page (the relay and the logs keep the short role ids).
const ROLE_NAMES = { metrics: "Metrics Analyst", changes: "Change Analyst", infra: "Infra Analyst", verifier: "Recovery Validator",
                     remediator: "Remediator", "incident-commander": "Commander" };
const who = (agent) => ROLE_NAMES[agent] || agent;
const roleOf = (agent) => (agent === "incident-commander" ? "commander" : agent);
const STEP_LABELS = { alert: "Alert", investigate: "Investigate", diagnosis: "Diagnose", approval: "Approve", restore: "Restore",
                      verify: "Verify", report: "Report" };
const STATE_TEXT = { done: "done", active: "in progress", waiting: "waiting for you", failed: "stopped", todo: "not yet" };
const AUDIT_TEXT = { allow: "ran", ask: "held for a human", approved: "ran after approval", deny: "refused by policy",
                     denied: "did not run: denied by a human", error: "error", approve: "approved by a human",
                     deny_decision: "denied by a human", connected: "connected by a human", cancel: "cancelled by a human" };
const TIER_TEXT = { allow: "runs at once", ask: "waits for a human to approve", deny: "never runs" };
const STARTING = new Set(["creating", "resuming"]);
const GONE = new Set(["deleting", "deleted"]);  // a specialist's MicroVM being deleted, or deleted
const MAX_STREAMS = 4;  // a browser opens about 6 connections to one site: the rest stays for the polls and decisions
const SOON_MS = 5 * 60 * 1000;  // the last minutes before an approval expires are shown in amber

const state = { token: "", selected: "", opened: "", list: [], incidents: new Map(), pendingKey: null, busy: false,
                deciding: new Set(),  // approval ids whose decision is in flight
                health: undefined, streams: new Set(), view: "incident", tab: "story", headlineOpen: false, policy: null,
                openEvents: new Set() };

function incidentState(incident) {  // what this tab has read of one incident
  if (!state.incidents.has(incident)) {
    state.incidents.set(incident, { seen: undefined, detail: null, logNext: 0, audit: [], activity: [], world: null,
                                    streamed: "", logTimer: 0, story: null, reportShown: false,
                                    logBusy: false, logAgain: false, auditKeys: new Set() });
  }
  return state.incidents.get(incident);
}

const itemOf = (incident) => state.list.find((entry) => entry.incident === incident);
// An incident whose commander's host is still starting (or failed to) has no host to read: the list says all there is.
const NO_HOST = new Set(["starting", "error"]);
const readable = (incident) => { const item = itemOf(incident); return !item || !NO_HOST.has(item.status); };

function readToken() {
  if (/(^#|&)(view=)?fleet\b/.test(location.hash)) state.view = "fleet";  // #fleet opens the fleet view (a second screen)
  const match = location.hash.match(/token=([^&]+)/);  // console.py demo --web passes it in the fragment, never sent to the server
  if (match) {
    sessionStorage.setItem("operatorToken", decodeURIComponent(match[1]));
    history.replaceState(null, "", location.pathname + (state.view === "fleet" ? "#fleet" : ""));
  }
  state.token = sessionStorage.getItem("operatorToken") || "";
  state.selected = state.selected || sessionStorage.getItem("selectedIncident") || "";
  $("token-form").hidden = Boolean(state.token);
  $("forget").hidden = !state.token;
  $("token-needed").hidden = Boolean(state.token);
  if (!state.token) $("empty").hidden = true;
}

function banner(text) {
  $("banner").textContent = text;
  $("banner").hidden = !text;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...(options.headers || {}), Authorization: `Bearer ${state.token}`, "Content-Type": "application/json" },
  });
  if (response.status === 401) {
    sessionStorage.removeItem("operatorToken");
    readToken();
    throw new Error("The operator token was refused.");
  }
  return response;
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function h(tag, className, ...children) {  // an element with children (nodes or text; false and null are skipped)
  const node = el(tag, undefined, className);
  for (const child of children.flat()) {
    if (child === false || child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

const clock = (seconds) => new Date(seconds * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit",
                                                                             hourCycle: "h23" });

function ago(seconds) {
  const s = Math.max(0, Math.round(Date.now() / 1000 - seconds));
  return s < 90 ? `${s} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${Math.round(s / 3600)} h`;
}

function phase(item) {
  // The incident's state as one tone for its dot: asking | working | done | failed | "" (idle).
  if (!item) return "";
  if (item.pending) return "asking";
  if (item.status === "busy") return "working";
  if (item.status === "error") return "failed";
  const report = (item.progress || []).find((p) => p.step === "report");
  return report && report.state === "done" ? "done" : "";
}

// -- the list: the relay's memory, every 3 s ---------------------------------------------------------------------------------

async function refreshList() {
  if (!state.token) return;
  const response = await api("incidents");
  if (!response.ok) throw new Error(`incidents: ${response.status}`);
  const { incidents } = await response.json();
  state.list = incidents;
  const ids = new Set(incidents.map((item) => item.incident));
  for (const id of state.incidents.keys()) if (!ids.has(id)) forget(id);
  $("empty").hidden = incidents.length > 0;
  for (const item of incidents) {
    const mine = incidentState(item.incident);
    const seen = `${item.status}/${item.pending}/${item.run || ""}`;
    const first = mine.seen === undefined;
    if (seen !== mine.seen) {
      mine.seen = seen;
      // The relay saw a change (no host was woken to find out): read the incident again. A first sighting reads nothing
      // (the selected incident is read by select), so opening the page does not wake every paused commander.
      if (!first && readable(item.incident)) refreshDetail(item.incident).catch((error) => banner(error.message));
    }
    if (item.status === "busy" && item.run && mine.streamed !== item.run) stream(item.incident, item.run);
  }
  if (state.selected && !ids.has(state.selected)) state.selected = "";
  if (!state.selected && incidents.length) state.selected = incidents[0].incident;
  if (state.selected && state.opened !== state.selected) select(state.selected, false);
  else render();
  if (!state.policy) refreshPolicy().catch(() => {});
}

function forget(incident) {
  const mine = state.incidents.get(incident);
  if (mine) clearInterval(mine.logTimer);
  state.incidents.delete(incident);
}

function select(incident, show = true) {
  if (incident !== state.opened) { state.pendingKey = null; state.headlineOpen = false; state.tab = "story"; }
  state.selected = state.opened = incident;
  if (show) state.view = "incident";
  sessionStorage.setItem("selectedIncident", incident);
  incidentState(incident);
  render();
  if (!readable(incident)) { state.opened = ""; return; }  // read once its host is up
  refreshDetail(incident).catch((error) => banner(error.message));
  refreshLog(incident).catch((error) => banner(error.message));
}

// -- one incident: its detail, its log, its runs ------------------------------------------------------------------------------

async function refreshDetail(incident) {
  if (!state.token || !incident || (state.busy && incident === state.selected)) return;
  const response = await api(`incidents/${encodeURIComponent(incident)}`);
  if (response.status === 404) {
    if (incident === state.selected) {
      banner(`Incident ${incident} is gone (its host was deleted).`);
      state.selected = "";
    }
    forget(incident);
    render();
    return;
  }
  if (!response.ok) throw new Error(`incident: ${response.status}`);
  const detail = await response.json();
  incidentState(incident).detail = detail;
  if (detail.story) takeStory(incident, detail.story);
  // The latest run's tool calls, replayed then live: for the selected incident (it was read anyway), and for any run that works.
  if (detail.run && (incident === state.selected || ["pending", "running"].includes(detail.run.status))) stream(incident, detail.run.run_id);
  render();
}

async function refreshLog(incident) {
  if (!state.token || !incident) return;
  const mine = incidentState(incident);
  // One read at a time: overlapping reads would each start at the same offset and add the same events. A read asked for
  // meanwhile is made by the one in flight, once it is done.
  if (mine.logBusy) { mine.logAgain = true; return; }
  mine.logBusy = true;
  try {
    do {
      mine.logAgain = false;
      for (let page = 0; page < 20; page += 1) {
        const response = await api(`incidents/${encodeURIComponent(incident)}/log?after=${mine.logNext}`);
        if (!response.ok || !state.incidents.has(incident)) return;
        const { events, next } = await response.json();
        events.forEach((event) => applyEvent(mine, event));
        const more = next > mine.logNext && events.length >= 500;
        mine.logNext = next;
        if (!more) break;
      }
    } while (mine.logAgain);
  } finally {
    mine.logBusy = false;
  }
  render();
}

function applyEvent(mine, event) {
  if (event.event === "world") { mine.world = event; return; }
  if (event.event === "gateway" || event.event === "decision") {
    const key = JSON.stringify([event.ts, event.event, event.tool, event.status || event.decision]);
    if (mine.auditKeys.has(key)) return;  // a log line already read
    mine.auditKeys.add(key);
    mine.audit.push({ ts: event.ts, kind: event.event, tone: event.event === "decision" ? event.decision : event.status,
                      tool: event.tool, error: event.error || "" });
    mine.audit = mine.audit.slice(-200);
  }
  // Agent host events: the relay keeps their states for the agent hosts in the list.
}

async function stream(incident, runId) {
  // The run's tool calls, replayed, then live. Several incidents stream side by side, up to MAX_STREAMS.
  const mine = incidentState(incident);
  if (mine.streamed === runId || state.streams.size >= MAX_STREAMS) return;
  mine.streamed = runId;
  state.streams.add(incident);
  try {
    const response = await api(`incidents/${encodeURIComponent(incident)}/runs/${encodeURIComponent(runId)}/events`);
    if (!response.ok || !response.body) return;
    clearInterval(mine.logTimer);
    mine.logTimer = setInterval(() => refreshLog(incident).catch(() => {}), 2000);  // only while a run works: the host is awake
    mine.activity.push({ kind: "start", run: runId });
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done || !state.incidents.has(incident)) break;
      buffer += value;
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const line = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        if (line.startsWith("data: ")) showEvent(incident, JSON.parse(line.slice(6)));
      }
    }
  } finally {
    state.streams.delete(incident);
    clearInterval(mine.logTimer);
    refreshLog(incident).catch((error) => banner(error.message));
    refreshDetail(incident).catch((error) => banner(error.message));
  }
}

function showEvent(incident, event) {
  const mine = incidentState(incident);
  if (event.type === "story") {  // a specialist reported, the verifier judged, or the postmortem was saved
    takeStory(incident, event.story);
    render();
    return;
  }
  mine.activity.push(event);
  mine.activity = mine.activity.slice(-400);
  if (incident === state.selected && state.view === "incident" && state.tab === "activity") renderActivity();
}

function takeStory(incident, story) {
  // The relay's story of an incident (from a detail read or the run's stream); the list's steps follow it at once.
  const mine = incidentState(incident);
  mine.story = story;
  const item = itemOf(incident);
  if (item && story.progress) item.progress = story.progress;
  if (story.report && !mine.reportShown) {  // the first time the postmortem is there, show it
    mine.reportShown = true;
    if (incident === state.selected) state.tab = "report";
  }
}

async function refreshPolicy() {
  const response = await api("policy");
  if (response.ok) { state.policy = await response.json(); renderPolicy(); }
}

async function refreshService() {
  if (!state.token || state.health === null) return;  // null: local mode, the log has it
  const response = await api("service");
  if (!response.ok) return;
  state.health = (await response.json()).health;
  render();
}

// -- rendering ---------------------------------------------------------------------------------------------------------------

function render() {
  renderTopbar();
  const fleet = state.view === "fleet";
  $("board").classList.toggle("fleet-mode", fleet);
  $("incident-view").hidden = fleet;
  $("fleet-view").hidden = !fleet;
  renderRail();
  if (fleet) renderFleet();
  else renderSelected();
}

function soonestExpiry(incident) {
  // The next approval of an incident to expire; an expired one only when none is left open (it can still be answered).
  const detail = (state.incidents.get(incident) || {}).detail;
  const times = ((detail && detail.pending) || []).map((stop) => Date.parse((stop.value || {}).expires_at)).filter((t) => !Number.isNaN(t));
  const open = times.filter((t) => t > Date.now());
  return open.length ? Math.min(...open) : times.length ? Math.max(...times) : Infinity;
}

function hostCount(item) {  // the commander's host and the specialists' hosts that exist now (one MicroVM each)
  const boxes = Object.values(item.hosts || {}).filter((tile) => tile && tile.status && !GONE.has(tile.status)).length;
  return 1 + boxes;
}

function renderTopbar() {
  const waiting = state.list.filter((item) => item.pending);
  const count = waiting.reduce((sum, item) => sum + item.pending, 0);
  const button = $("decisions-waiting");
  document.title = count ? `(${count}) Incident response on MicroVMs` : "Incident response on MicroVMs";
  button.classList.toggle("on", count > 0);
  button.disabled = count === 0;
  if (!count) {
    button.textContent = "No decision waiting";
    delete button.dataset.incident;
  } else {
    const rank = (item) => { const at = soonestExpiry(item.incident); return at > Date.now() ? at : Number.MAX_SAFE_INTEGER; };
    const first = [...waiting].sort((a, b) => rank(a) - rank(b))[0];
    const at = soonestExpiry(first.incident);
    const left = Number.isFinite(at) ? ` · ${expiryOf(new Date(at).toISOString()).text.replace(/ \(at .*\)$/, "")}` : "";
    button.textContent = `⚑ ${count} decision${count > 1 ? "s" : ""} waiting · ${first.incident}${left} · Open`;
    button.dataset.incident = first.incident;
  }
  $("view-incident").setAttribute("aria-pressed", String(state.view === "incident"));
  $("view-fleet").setAttribute("aria-pressed", String(state.view === "fleet"));
  $("fleet-count").textContent = state.list.length ? String(state.list.reduce((sum, item) => sum + hostCount(item), 0)) : "";
}

function stepNow(progress) {
  // The step the incident is at (the first not done), as [index, state]; [last, "done"] when all are done.
  const steps = progress || [];
  const i = steps.findIndex((p) => p.state !== "done");
  return i < 0 ? [Math.max(steps.length - 1, 0), steps.length ? "done" : "todo"] : [i, steps[i].state];
}

function stepWords(item) {
  const steps = item.progress || [];
  if (!steps.length) return h("span", "rail-step", item.status === "error" ? "failed to start" : "starting");
  const [i, where] = stepNow(steps);
  const verdict = item.verdict;
  const word = where === "done" ? (verdict === "not_healthy" ? "Ended: not healthy" : "Resolved")
    : steps.some((p) => p.state === "failed") && !steps.some((p) => p.state === "active" || p.state === "waiting") ? "Ended"
      : STEP_LABELS[steps[i].step] || steps[i].step;
  return h("span", "rail-step", h("b", "", word), ` · step ${i + 1} of ${steps.length}`);
}

function strip(item) {
  // The agent hosts in one line: the commander's host, then one dot per specialist's host.
  const line = h("span", "strip");
  line.setAttribute("aria-label", "agent hosts");
  const [look] = commanderLook(item);
  line.append(el("i", undefined, `c ${look === "running" || look === "starting" ? "on" : look === "asking" ? "ask" : look === "failed" ? "bad" : "pa"}`));
  for (const role of ROLES) {
    const [boxLook] = hostLook(item, role);
    line.append(el("i", undefined, boxLook === "running" || boxLook === "starting" ? "on" : boxLook === "paused" ? "pa" : ""));
  }
  return line;
}

function renderRail() {
  const list = $("rail");
  list.replaceChildren();
  for (const item of state.list) {
    const row = el("button", undefined, `rail-row${item.pending ? " asks" : ""}`);
    row.type = "button";
    row.setAttribute("aria-current", String(item.incident === state.selected));
    row.append(h("span", "rail-top", el("span", undefined, `dot ${phase(item)}`), el("span", item.incident, "rail-id"),
                 item.pending ? el("span", "⚑ decide", "flag") : null),
               stepWords(item), strip(item));
    row.addEventListener("click", () => select(item.incident));
    const li = el("li");
    li.append(row);
    list.append(li);
  }
}

// -- the agent hosts: one lane per incident -------------------------------------------------------------------------------

function commanderLook(item) {
  // [look, state, note] of the commander's agent host: asking | running | starting | failed | paused.
  if (item.pending) return ["asking", "a specialist waits for you", "the approval waits in the Remediator's host"];
  if (item.status === "busy") {
    const last = item.last_tool && roleOf(item.last_tool.agent) === "commander" ? `→ ${item.last_tool.tool}` : "delegating";
    return ["running", "running", last];
  }
  if (item.status === "error") return ["failed", item.run ? "run failed" : "start failed", item.error || ""];
  if (!item.status || item.status === "starting") return ["starting", "starting", item.host ? `MicroVM ${item.host}` : ""];
  const report = (item.progress || []).find((p) => p.step === "report");
  return ["paused", report && report.state === "done" ? "done" : "idle", item.host ? `MicroVM ${item.host}` : ""];
}

function hostLook(item, role) {
  // [look, state, note] of one specialist's agent host: running | starting | paused | failed | none.
  const tile = (item.hosts || {})[role];
  if (!tile || !tile.status) return ["none", "—", item.status === "busy" ? "not started yet" : "not needed yet"];
  if (GONE.has(tile.status)) return ["none", tile.status, tile.note || ""];
  const look = STARTING.has(tile.status) ? "starting" : ["paused", "failed"].includes(tile.status) ? tile.status : "running";
  return [look, tile.status, [tile.note, `for ${ago(tile.since)}`].filter(Boolean).join(" · ")];
}

function tile(classes, kind, name, status, note) {
  return h("div", `tile ${classes}`, el("span", kind, "kind"), el("span", name, "name"), el("span", status, "st"), el("span", note, "note"));
}

function lane(item, compact) {
  const [look, status, note] = commanderLook(item);
  const commander = tile(`commander ${look}`, "Commander · agent host", "Commander", status, note);
  const row = h("div", `lane${compact ? " compact" : ""}`, commander);
  const link = el("span", undefined, "link");
  link.setAttribute("aria-hidden", "true");
  row.append(link);
  for (const role of ROLES) {
    const [boxLook, boxStatus, boxNote] = hostLook(item, role);
    row.append(tile(`r-${role} ${boxLook}`, "Specialist · agent host", who(role), boxStatus, boxNote));
  }
  return row;
}

// -- the incident view --------------------------------------------------------------------------------------------------------

function serviceOf(incident) {
  // The service as last read: the relay's own reading (real mode), else the newest one in this incident's log (local mode).
  const mine = state.incidents.get(incident);
  return state.health || (mine && mine.world) || null;
}

function serviceName(item) {  // the alert's resource; health.droplet is the droplet's state (active), not its name
  return (item && item.alarm && item.alarm.resource) || "the service";
}

function serviceBits(item, health, full) {
  const bits = [];
  if (full) bits.push(el("span", "Service", "label"));
  bits.push(el("span", serviceName(item), "mono"));
  if (health || full) bits.push(el("span", health ? String(health.status) : "no reading yet", `pill ${health ? health.status : ""}`));
  if (full && health) {
    const facts = [health.version ? `version ${shortVersion(health.version)}` : "",
                   typeof health.error_rate === "number" ? `${(health.error_rate * 100).toFixed(0)} % errors` : "",
                   health.droplet && health.droplet !== "active" ? `droplet ${health.droplet}` : "",
                   typeof health.ts === "number" ? `read ${ago(health.ts)} ago` : ""].filter(Boolean);
    if (facts.length) bits.push(el("span", facts.join(" · "), "muted"));
  }
  return bits;
}

const shortVersion = (v) => (/^[0-9a-f]{12,}$/.test(String(v)) ? String(v).slice(0, 7) : String(v));

function alarmText(alarm) {
  const value = typeof alarm.value === "number" ? ` ${alarm.value.toFixed(0)}` : alarm.value ? ` ${alarm.value}` : "";
  return `${alarm.rule || "alert"}: ${alarm.metric || "metric"}${value}${alarm.resource ? ` on ${alarm.resource}` : ""}`;
}

function renderSelected() {
  const mine = state.selected && state.incidents.get(state.selected);
  const item = itemOf(state.selected);
  $("detail").hidden = !mine || !item;
  if (!mine || !item) return;
  const detail = mine.detail;
  $("detail-title").textContent = state.selected;
  const alarm = item.alarm;
  $("alarm").hidden = !alarm;
  if (alarm) {
    $("alarm").replaceChildren(el("span", "ALARM", "tag"), alarmText(alarm), el("span", ` · fired ${ago(alarm.at)} ago`, "muted"));
  }
  const idle = detail && detail.status === "idle" && detail.pending.length === 0;
  $("close-incident").hidden = !idle;  // the relay refuses during a run
  $("service").replaceChildren(...serviceBits(item, serviceOf(state.selected), true));
  const steps = (mine.story && mine.story.progress) || item.progress || [];
  $("steps").replaceChildren(...steps.map(({ step, state: where }, i) => {
    const li = el("li", STEP_LABELS[step] || step, `step ${where}`);
    li.dataset.n = String(i + 1);
    li.title = `${STEP_LABELS[step] || step}: ${STATE_TEXT[where] || where}`;
    return li;
  }));
  renderHeadline(detail);
  if (detail) {
    $("message-form").hidden = !idle;
    const key = JSON.stringify([detail.incident, detail.pending.map((stop) => [stop.id, stop.same_elsewhere || []])]);
    if (key !== state.pendingKey) {  // re-render only when the approvals change: a reason being typed survives
      state.pendingKey = key;
      renderApprovals(detail.pending);
    }
  } else {
    $("approvals").replaceChildren();
  }
  renderOutcome(mine.story, steps);
  const hosts = hostCount(item);
  $("runtime-count").textContent = `${hosts} MicroVM${hosts === 1 ? "" : "s"} · ${runningCount(item)} running`;
  $("lane").replaceChildren(lane(item, false));
  renderTabs(mine);
}

function runningCount(item) {
  const [look] = commanderLook(item);
  return (look === "running" ? 1 : 0) + ROLES.filter((role) => ["running", "starting"].includes(hostLook(item, role)[0])).length;
}

function renderHeadline(detail) {
  const text = (detail && detail.last_message) || "";
  $("headline").textContent = detail ? (text ? summaryOf(text) : "Nothing said yet.") : "Reading the incident…";
  $("headline-more").hidden = !text;
  $("headline-more").textContent = state.headlineOpen ? "Hide the full message" : "Full message";
  $("headline-more").setAttribute("aria-expanded", String(state.headlineOpen));
  $("last-message").hidden = !state.headlineOpen || !text;
  if (state.headlineOpen && text) $("last-message").replaceChildren(markdown(text, "md"));
}

function renderOutcome(story, steps) {
  const box = $("outcome");
  const verdict = story && story.verdict && story.verdict.verdict ? story.verdict : null;
  const denied = steps.some((p) => p.step === "approval" && p.state === "failed");
  box.hidden = !verdict && !denied;
  if (box.hidden) return;
  const good = verdict && verdict.verdict === "healthy";
  box.className = `outcome${good ? "" : " bad"}`;
  const facts = verdict ? [verdict.reason, verdict.version ? `version ${shortVersion(verdict.version)}` : "",
                           typeof verdict.error_rate === "number" ? `${(verdict.error_rate * 100).toFixed(0)} % errors` : ""].filter(Boolean) : [];
  const check = verdict && verdict.relay_check;
  const second = check ? ` The relay's own reading of the service: ${check.verdict === "healthy" ? "healthy" : "not healthy"}${check.agrees ? "." : ", which differs from the verdict: check the service before you rely on it."}` : "";
  if (check && !check.agrees) box.className = "outcome bad";
  box.replaceChildren(el("span", verdict ? (good ? "Healthy" : "Not healthy") : "Restore denied", "pill"),
                      el("span", verdict ? `The Recovery Validator's verdict${facts.length ? `: ${facts.join(" · ")}` : "."}${second}`
                                         : "A human denied the restore: it did not run, nothing was changed."));
}

// -- tabs -------------------------------------------------------------------------------------------------------------------

const TABS = ["story", "activity", "gateway", "report"];

function renderTabs(mine) {
  const report = mine.story && mine.story.report;
  $("tab-report").hidden = !report;
  if (state.tab === "report" && !report) state.tab = "story";
  for (const name of TABS) {
    $(`tab-${name}`).setAttribute("aria-selected", String(state.tab === name));
    $(`panel-${name}`).hidden = state.tab !== name;
  }
  const events = storyEvents(state.selected);
  $("n-story").textContent = String(events.length);
  $("n-activity").textContent = String(mine.activity.filter((event) => event.tool).length);
  $("n-gateway").textContent = String(mine.audit.length);
  if (state.tab === "story") renderTimeline(events);
  if (state.tab === "activity") renderActivity();
  if (state.tab === "gateway") { renderPolicy(); renderAudit(mine); }
  if (state.tab === "report" && report) {
    $("report").replaceChildren(el("h3", report.title, "report-title"), markdown(report.body, "md"));
  }
}

function auditStory(row) {
  // A gateway or decision row as a line of the story; an allowed read is left to the Action Gateway tab.
  const tool = row.tool || "an action";
  if (row.kind === "decision") {
    const yes = row.tone === "approve" || row.tone === "connected";
    return { who: "You", role: yes ? "human" : "deny", text: yes ? `Approved ${tool}.` : `Denied ${tool}: it did not run.` };
  }
  if (row.tone === "ask") return { who: "Action Gateway", role: "ask", text: `Held ${tool} for a human (policy tier ask).` };
  if (row.tone === "approved") return { who: "Action Gateway", role: "ok", text: `${tool} ran after approval.` };
  if (row.tone === "deny") return { who: "Action Gateway", role: "deny", text: `Refused ${tool} (policy tier deny): it never runs.` };
  if (row.tone === "error") return { who: "Action Gateway", role: "deny", text: `${tool} failed${row.error ? `: ${row.error}` : "."}` };
  return null;
}

function sentGroups(sent) {
  // The sub-agents the commander sent, the ones sent within a few seconds of each other as one group (sent in parallel).
  const groups = [];
  for (const one of [...sent].sort((a, b) => a.at - b.at)) {
    const last = groups[groups.length - 1];
    if (last && one.at - last[last.length - 1].at <= 5) last.push(one);
    else groups.push([one]);
  }
  return groups;
}

function sentText(roles) {
  const names = roles.map(who);
  const list = names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
  if (roles.every((role) => role === "remediator")) return "Started the Remediator in its own agent host: its Action Gateway asks a human before it acts.";
  if (roles.length > 1) return `Started the ${list}, each in its own agent host, in parallel.`;
  return `Started the ${list} in its own agent host.`;
}

function storyEvents(incident) {
  const item = itemOf(incident);
  const mine = state.incidents.get(incident);
  const story = (mine && mine.story) || {};
  const events = [];
  if (item && item.alarm) events.push({ at: item.alarm.at, who: "Alert", role: "alert", text: `${alarmText(item.alarm)}. A commander started in its own agent host.` });
  for (const group of sentGroups(story.sent || [])) {
    events.push({ at: group[0].at, who: "Commander", role: "commander", text: sentText(group.map((sent) => sent.role)) });
  }
  for (const finding of story.findings || []) {
    events.push({ at: finding.at, who: who(finding.role), role: finding.role, text: summaryOf(finding.text), full: finding.text });
  }
  for (const row of (mine && mine.audit) || []) {
    const line = auditStory(row);
    if (line) events.push({ at: row.ts, policy: true, ...line });
  }
  if (story.verdict && story.verdict.verdict) {
    const good = story.verdict.verdict === "healthy";
    events.push({ at: story.verdict.at, who: "Verdict", role: good ? "ok" : "deny",
                  text: `${good ? "Healthy" : "Not healthy"}${story.verdict.reason ? `: ${story.verdict.reason}` : "."}${
                    story.verdict.relay_check ? ` The relay's own reading: ${story.verdict.relay_check.verdict === "healthy" ? "healthy" : "not healthy"}${story.verdict.relay_check.agrees ? "" : " (differs)"}.` : ""}` });
  }
  if (story.report) events.push({ at: story.report.at, who: "Commander", role: "commander", text: `Saved the postmortem: ${story.report.title}` });
  return events.sort((a, b) => (b.at || 0) - (a.at || 0));
}

function renderTimeline(events) {
  const list = $("timeline");
  list.replaceChildren();
  if (!events.length) list.append(el("li", "Nothing yet. Each sub-agent's report appears here as it comes back to the commander.", "muted"));
  for (const event of events) {
    const li = el("li", undefined, `ev r-${event.role}${event.policy ? " policy" : ""}`);
    const time = el("time", event.at ? clock(event.at) : "");
    const mark = el("span", undefined, "mark-line");
    const body = h("div", "body", el("span", event.who, "who"), el("p", event.text, "txt"));
    if (event.full) {
      const key = `${event.role}-${event.at}`;
      const more = el("details");
      more.open = state.openEvents.has(key);
      more.append(el("summary", "Full report"), markdown(event.full, "md"));
      more.addEventListener("toggle", () => { if (more.open) state.openEvents.add(key); else state.openEvents.delete(key); });
      body.append(more);
    }
    li.append(time, mark, body);
    list.append(li);
  }
}

function renderActivity() {
  const mine = state.incidents.get(state.selected);
  const body = $("activity");
  body.replaceChildren();
  const events = [...((mine && mine.activity) || [])].reverse();
  for (const event of events) {
    const tr = el("tr");
    if (event.kind === "start") tr.append(Object.assign(el("td", `run ${event.run}`, "end"), { colSpan: 2 }));
    else if (event.type === "end") tr.append(Object.assign(el("td", `run ended: ${event.status}${event.pending ? ` (${event.pending} to decide)` : ""}`, "end"), { colSpan: 2 }));
    else {
      const input = JSON.stringify(event.input ?? {});
      tr.append(el("td", who(event.agent), `who r-${roleOf(event.agent)}`), el("td", `${event.tool} ${input.length > 160 ? `${input.slice(0, 159)}…` : input}`, "m"));
    }
    body.append(tr);
  }
  if (!events.length) body.append(h("tr", "", el("td", "No tool call seen yet.", "muted")));
}

function renderPolicy() {
  const body = $("policy");
  body.replaceChildren();
  if (!state.policy) { body.append(h("tr", "", el("td", "Reading the policy…", "muted"))); return; }
  for (const tier of ["allow", "ask", "deny"]) {
    const tools = state.policy[tier] || [];
    body.append(h("tr", "", h("td", "", el("span", tier, `tier ${tier}`)), el("td", TIER_TEXT[tier]),
                  el("td", tools.length ? tools.join(", ") : "none", "m")));
  }
}

function auditTier(row) {
  if (row.kind === "decision") return "ask";
  return { allow: "allow", ask: "ask", approved: "ask", denied: "ask", deny: "deny" }[row.tone] || "";
}

function renderAudit(mine) {
  const body = $("audit");
  body.replaceChildren();
  for (const row of [...mine.audit].reverse()) {
    const tier = auditTier(row);
    const outcome = row.kind === "decision" ? AUDIT_TEXT[row.tone === "deny" ? "deny_decision" : row.tone] || row.tone
      : row.error || AUDIT_TEXT[row.tone] || row.tone;
    body.append(h("tr", "", el("td", clock(row.ts), "t"), h("td", "", tier ? el("span", tier, `tier ${tier}`) : ""),
                  el("td", row.tool, "m"), el("td", String(outcome).slice(0, 200))));
  }
  if (!mine.audit.length) body.append(h("tr", "", el("td", "No gateway call yet.", "muted")));
}

// -- approvals -------------------------------------------------------------------------------------------------------------

function expiryOf(iso, now = Date.now()) {
  // "in 23 min (at 21:53)", "in 40 s (at 21:53)", "expired 2 min ago (at 21:53)"; tone ok | soon | expired.
  const at = Date.parse(iso);
  if (Number.isNaN(at)) return { text: iso || "not given", tone: "" };
  const hm = new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
  const left = at - now;
  const span = (ms) => (Math.abs(ms) < 60000 ? `${Math.round(Math.abs(ms) / 1000)} s` : `${Math.round(Math.abs(ms) / 60000)} min`);
  if (left <= 0) return { text: `expired ${span(left)} ago (at ${hm})`, tone: "expired" };
  return { text: `in ${span(left)} (at ${hm})`, tone: left <= SOON_MS ? "soon" : "ok" };
}

function showExpiry(card) {
  const cell = card.querySelector(".expires");
  if (!card.dataset.expires) { cell.textContent = ""; return; }
  const { text, tone } = expiryOf(card.dataset.expires);
  cell.textContent = tone === "expired" ? text : `expires ${text}`;
  cell.className = `expires ${tone}`;
  card.querySelector(".expired-note").hidden = tone !== "expired";
  card.classList.toggle("expired", tone === "expired");
}

function showSameElsewhere(card, others) {
  // Another open incident with the same tool and arguments: two restores of one droplet, for example.
  const list = card.querySelector(".same-elsewhere");
  list.replaceChildren();
  for (const other of others) {
    const at = new Date(other.at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
    list.append(el("li", other.state === "approved"
      ? `Incident ${other.incident} already approved this same action (at ${at}).`
      : `Incident ${other.incident} is also waiting for a decision on this same action.`));
  }
  list.hidden = others.length === 0;
}

function named(facts, label) {
  // The name in a fact the relay read ("droplet    dhr-checkout-svc (active)"); null when it is not there or not found.
  const line = (facts || []).find((fact) => fact.startsWith(label));
  if (!line || /NOT FOUND/.test(line)) return null;
  const match = line.slice(label.length).trim().match(/^([^\s,(]+)/);
  return match ? match[1] : null;
}

function question(value, facts) {
  // The approval in plain words; the tool and its arguments stay under Details.
  const tool = String(value.tool || "").replace(/@\d+$/, "");
  const args = value.arguments || {};
  if (value.type === "connect_required") return `Connect an account so the commander can use ${tool}?`;
  if (tool === "digitalocean_restore-droplet") {
    return `Restore droplet ${named(facts, "droplet") || args.ID} from snapshot ${named(facts, "snapshot") || args.ImageID}?`;
  }
  if (tool === "digitalocean_droplet-delete") return `Delete droplet ${named(facts, "droplet") || args.ID}?`;
  return `Run ${tool}?`;
}

function firstSentence(text) {
  const flat = plain(String(text || ""));
  const match = flat.match(/^.{20,}?[.!?](\s|$)/);
  return match ? match[0].trim() : flat;
}

function renderApprovals(pending) {
  const box = $("approvals");
  box.replaceChildren();
  for (const stop of pending) {
    const value = stop.value || {};
    const card = $("approval-template").content.firstElementChild.cloneNode(true);
    const connect = value.type === "connect_required";
    card.querySelector(".question").textContent = question(value, stop.facts);
    card.querySelector(".evidence").textContent = firstSentence(value.rationale) || "the commander gave no reason";
    card.querySelector(".evidence-full").textContent = value.rationale || "";
    card.querySelector(".arguments").textContent = `${value.tool || ""}\n${JSON.stringify(value.arguments || {}, null, 2)}`;
    const checked = card.querySelector(".checked");
    for (const fact of stop.facts || []) checked.append(el("li", fact.replace(/\s{2,}/g, " ")));
    for (const problem of stop.problems || []) checked.append(el("li", `Cannot be approved: ${problem}`, "bad"));
    if (!(stop.facts || []).length && !(stop.problems || []).length) checked.append(el("li", "nothing read for this action", "muted"));
    showSameElsewhere(card, stop.same_elsewhere || []);
    if (value.expires_at) card.dataset.expires = value.expires_at;
    showExpiry(card);
    if (connect) {
      card.querySelector(".held").hidden = true;
      card.querySelector(".why").hidden = !value.rationale;
      card.querySelector(".connect").hidden = false;
      card.querySelector(".reason").hidden = true;
      const link = card.querySelector(".link");
      link.textContent = value.url || "";
      if (typeof value.url === "string" && value.url.startsWith("https://")) link.href = value.url;  // never a javascript: URL
    }
    const actions = card.querySelector(".actions");
    const choices = connect ? [["connected", "I connected it", "approve"], ["cancel", "Cancel", "deny"]]
                            : [["approve", "Approve", "approve"], ["deny", "Deny", "deny"]];
    for (const [decision, label, style] of choices) {
      const button = el("button", label, style);
      button.type = "button";
      if (decision === "approve" && (stop.problems || []).length) button.disabled = true;  // the relay would refuse it
      if (state.deciding.has(stop.id)) button.disabled = true;  // a decision on it is in flight
      button.addEventListener("click", () => decide(stop.id, decision, card));
      actions.append(button);
    }
    box.append(card);
  }
}

async function decide(interruptId, decision, card) {
  const incident = state.selected;
  state.busy = true;
  card.querySelectorAll("button").forEach((button) => { button.disabled = true; });
  const reason = card.querySelector(".reason").value.trim();
  let taken = false;
  state.deciding.add(interruptId);  // a redraw of the approvals keeps its buttons disabled until the outcome is known
  try {
    const response = await api(`incidents/${encodeURIComponent(incident)}/decision`, {
      method: "POST", body: JSON.stringify({ interrupt_id: interruptId, decision, ...(reason ? { reason } : {}) }),
    });
    if (response.status === 409) banner("Already decided (by you or another approver), or the thread moved on.");
    else if (!response.ok) banner(`The decision was not taken: ${response.status} ${await response.text()}`);
    else {
      taken = true;
      banner(`${decision}: sent. The specialist that asked continues in its agent host.`);
      state.tab = "story";
      stream(incident, (await response.json()).run_id);
    }
  } finally {
    state.busy = false;
    state.pendingKey = null;
    if (!taken) state.deciding.delete(interruptId);  // refused or failed: it can be decided again
    await refreshDetail(incident).catch((error) => banner(error.message));
    state.deciding.delete(interruptId);  // taken: the detail just read no longer lists it
  }
}

async function closeIncident() {
  const incident = state.selected;
  if (!incident || !window.confirm(`Close incident ${incident}? Its MicroVMs (the commander's agent host and its specialists') are deleted, with its thread and event log.`)) return;
  const button = $("close-incident");
  button.disabled = true;
  state.busy = true;
  try {
    const response = await api(`incidents/${encodeURIComponent(incident)}`, { method: "DELETE" });
    if (response.status === 409) banner("Not closed: a run is in flight. Wait for it to end.");
    else if (!response.ok) banner(`Not closed: ${response.status} ${await response.text()}`);
    else {
      banner(`Incident ${incident} closed: deleted ${(await response.json()).deleted.join(", ")}.`);
      state.selected = "";
      forget(incident);
    }
  } finally {
    button.disabled = false;
    state.busy = false;
    await refreshList().catch((error) => banner(error.message));
  }
}

// -- the fleet view ------------------------------------------------------------------------------------------------------------

function renderFleet() {
  const list = $("fleet");
  list.replaceChildren();
  let specialists = 0;
  let running = 0;
  for (const item of state.list) {
    specialists += hostCount(item) - 1;
    running += runningCount(item);
    const row = el("li", undefined, `frow${item.pending ? " asks" : ""}`);
    const open = el("button", "Open ›", "open");
    open.type = "button";
    open.addEventListener("click", () => select(item.incident));
    row.addEventListener("click", (event) => { if (!event.target.closest("button")) select(item.incident); });
    const health = serviceOf(item.incident);
    row.append(h("div", "itile",
                 h("span", "rail-top", el("span", undefined, `dot ${phase(item)}`), el("span", item.incident, "rail-id"),
                   item.pending ? el("span", "⚑ decide", "flag") : null),
                 h("span", "svc", ...serviceBits(item, health, false)),
                 stepWords(item),
                 item.alarm ? el("span", `alarm fired ${ago(item.alarm.at)} ago`, "muted") : null,
                 open),
               h("div", "fright", lane(item, true), saidOf(item)));
    list.append(row);
  }
  if (!state.list.length) list.append(el("li", "No incident open.", "empty muted"));
  $("stat-incidents").textContent = String(state.list.length);
  $("stat-hosts").textContent = String(state.list.length);
  $("stat-specialists").textContent = String(specialists);
  $("stat-running").textContent = String(running);
}

function saidOf(item) {
  const detail = (state.incidents.get(item.incident) || {}).detail;
  const line = h("p", "said", el("b", "Commander: "));
  if (detail && detail.last_message) line.append(summaryOf(detail.last_message));
  else if (item.last_tool) line.append(`${who(item.last_tool.agent)} → ${item.last_tool.tool}`);
  else line.append("nothing yet");
  return line;
}

// -- text helpers: a one-line summary, and a small Markdown reader --------------------------------------------------------------

function plain(text) {  // a line of Markdown as plain text, for a one-line summary
  return text.replace(/[*`#>|]/g, "").replace(/\s+/g, " ").trim();
}

function summaryOf(text) {
  // The first line that says something: not a heading, not a pleasantry ("Perfect! I now have…").
  // Labels ("Droplet Status:"), table rows and headings are skipped; a line naming a cause, a suspect, an onset, a snapshot, a
  // recommendation or a verdict wins over the first sentence.
  const candidates = [];
  for (const line of String(text).split("\n")) {
    const trimmed = line.trim();
    const flat = plain(line).replace(/^(\d+[.)]|[-•])\s*/, "");
    if (flat.length < 25 || /^[#|]/.test(trimmed) || /:$/.test(flat)) continue;
    if (/^(perfect|great|excellent|now|i (now )?have|i've|here('s| is| are)|based on)\b/i.test(flat)) continue;
    candidates.push(flat);
  }
  const telling = /\b(root cause|cause[sd]?|suspect|culprit|onset|started|predates|before the|recommend|verdict|restor)/i;
  const pick = candidates.find((line) => telling.test(line)) || candidates[0] || plain(String(text));
  return pick.length > 220 ? `${pick.slice(0, 219)}…` : pick;
}

function inline(text) {
  // **bold** and `code` as elements, everything else as text nodes.
  const frag = document.createDocumentFragment();
  for (const part of String(text).split(/(\*\*[^*]+\*\*|`[^`]+`)/)) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) frag.append(el("strong", part.slice(2, -2)));
    else if (part.startsWith("`") && part.endsWith("`") && part.length > 2) frag.append(el("code", part.slice(1, -1)));
    else frag.append(document.createTextNode(part));
  }
  return frag;
}

function markdown(text, className) {
  // A small Markdown reader for the agents' reports: headings, lists, tables, code blocks, paragraphs. Every piece of text is
  // put in with textContent (through el and inline); nothing is parsed as HTML.
  const root = el("div", undefined, className);
  const lines = String(text || "").split("\n");
  let list = null;
  let para = [];
  const flush = () => {
    if (para.length) {
      const p = el("p");
      p.append(inline(para.join(" ")));
      root.append(p);
      para = [];
    }
  };
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();
    if (trimmed.startsWith("```")) {
      flush(); list = null;
      const code = [];
      for (i += 1; i < lines.length && !lines[i].trim().startsWith("```"); i += 1) code.push(lines[i]);
      root.append(el("pre", code.join("\n")));
      continue;
    }
    if (trimmed.startsWith("|")) {
      flush(); list = null;
      const table = el("table");
      for (; i < lines.length && lines[i].trim().startsWith("|"); i += 1) {
        const cells = lines[i].trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
        if (cells.every((cell) => /^:?-{2,}:?$/.test(cell))) continue;  // the separator row
        const tr = el("tr");
        for (const cell of cells) {
          const td = el(table.rows.length ? "td" : "th");
          td.append(inline(cell));
          tr.append(td);
        }
        table.append(tr);
      }
      i -= 1;
      root.append(table);
      continue;
    }
    const heading = trimmed.match(/^(#{1,6})\s+(.*)$/);
    const bullet = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (heading) {
      flush(); list = null;
      const hd = el("p", undefined, `md-h md-h${Math.min(heading[1].length, 3)}`);
      hd.append(inline(heading[2]));
      root.append(hd);
    } else if (bullet) {
      flush();
      if (!list) { list = el("ul"); root.append(list); }
      const li = el("li");
      li.append(inline(bullet[1]));
      list.append(li);
    } else if (!trimmed || /^-{3,}$/.test(trimmed)) {
      flush(); list = null;
    } else {
      list = null;
      para.push(trimmed);
    }
  }
  flush();
  return root;
}

// -- controls and timers ---------------------------------------------------------------------------------------------------------

$("token-form").addEventListener("submit", (event) => {
  event.preventDefault();
  sessionStorage.setItem("operatorToken", $("token").value.trim());
  $("token").value = "";
  readToken();
  banner("");
  refreshList().catch((error) => banner(error.message));
  refreshService().catch(() => {});
});
$("forget").addEventListener("click", () => { sessionStorage.removeItem("operatorToken"); readToken(); });
$("close-incident").addEventListener("click", () => closeIncident());
$("view-incident").addEventListener("click", () => { state.view = "incident"; render(); });
$("view-fleet").addEventListener("click", () => { state.view = "fleet"; render(); });
$("decisions-waiting").addEventListener("click", () => {
  if ($("decisions-waiting").dataset.incident) select($("decisions-waiting").dataset.incident);
});
$("headline-more").addEventListener("click", () => { state.headlineOpen = !state.headlineOpen; render(); });
for (const name of TABS) $(`tab-${name}`).addEventListener("click", () => { state.tab = name; render(); });
$("message-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const incident = state.selected;
  const text = $("message").value.trim();
  const response = await api(`incidents/${encodeURIComponent(incident)}/messages`, { method: "POST", body: JSON.stringify({ text }) });
  if (!response.ok) { banner(`Not sent: ${response.status} ${await response.text()}`); return; }
  $("message").value = "";
  stream(incident, (await response.json()).run_id);
});

readToken();
// The list comes from the relay's memory: polling it wakes no host. An incident is read again only when the list says it
// changed, so a commander waiting for a decision can pause in its host.
const tick = () => refreshList().catch((error) => banner(error.message));
tick();
setInterval(tick, 3000);
setInterval(() => {  // the countdowns and ages, without a request
  document.querySelectorAll(".decision").forEach(showExpiry);
  render();
}, 5000);
refreshService().catch(() => {});
setInterval(() => refreshService().catch(() => {}), 5000);
