// Browser client for /ws/live and /api/*. Protocol: see src/server/live.py.
"use strict";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
let token = params.get("token") || "";
try { if (token) sessionStorage.setItem("swi-token", token); else token = sessionStorage.getItem("swi-token") || ""; } catch (e) {}
if (params.has("token")) history.replaceState(null, "", location.pathname);   // keep the token out of the address bar

const authHeaders = () => (token ? { Authorization: "Bearer " + token } : {});
const withToken = (url) => (token ? url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token) : url);

function log(msg) {
  const el = $("log");
  el.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  el.scrollTop = el.scrollHeight;
}
function setTurn(text, cls) { const t = $("turn"); t.textContent = text; t.className = "turn " + cls; }
const fmtS = (ms) => (ms == null ? "–" : (ms / 1000).toFixed(1) + " s");

// ---------------------------------------------------------------- server status
let status = null;
async function refreshStatus() {
  try {
    const r = await fetch("/api/status", { headers: authHeaders() });
    if (r.status === 401) { $("model-status").textContent = "Token required — open the link printed by 'serve'"; $("model-status").className = "pill bad"; return; }
    status = await r.json();
  } catch (e) { $("model-status").textContent = "Server unreachable"; $("model-status").className = "pill bad"; return; }
  const p = status.pool;
  const pill = $("model-status");
  pill.textContent = p.state === "ready" ? (p.busy ? "Models ready · in use" : "Models ready")
                   : p.state === "loading" || p.state === "idle" ? "Loading models…" : "Models failed: " + (p.error || "");
  pill.className = "pill " + (p.state === "ready" ? "ok" : p.state === "failed" ? "bad" : "warn");
  $("consent").textContent = status.consent;
  $("upload-limits").textContent = `Up to ${status.limits.max_upload_mb} MB / ${Math.round(status.limits.max_upload_s / 60)} min`;
  const sys = {
    Profile: status.profile,
    "Speech recognition": `${status.asr.model} (beam ${status.asr.beam_size}${status.asr.corrections ? ", correction list on" : ""})`,
    Translation: `${status.mt.model} (${status.mt.license})`,
    "English voice": `${status.tts.backend}${status.tts.voice ? " · " + status.tts.voice : ""}`,
    "Model load": p.load_ms ? fmtS(p.load_ms) : "–",
    "Session limit": `${Math.round(status.limits.max_session_s / 60)} min, idle ${status.limits.idle_timeout_s} s`,
  };
  if (status.session) {
    const q = status.session.queues;
    sys["Live queues"] = `audio ${q.audio} · mt ${q.mt} · tts ${q.tts} (max ${q.max})`;
  }
  $("system").innerHTML = Object.entries(sys).map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("");
  const free = p.state !== "failed" && !p.busy;
  if (!live) { $("start-btn").disabled = !free; $("upload-btn").disabled = !free; }
}
function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

// ---------------------------------------------------------------- devices
async function listDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devs = await navigator.mediaDevices.enumerateDevices();
  for (const [sel, kind] of [[$("mic-select"), "audioinput"], [$("out-select"), "audiooutput"]]) {
    const keep = sel.value;
    sel.length = 1;
    devs.filter((d) => d.kind === kind && d.deviceId && d.deviceId !== "default").forEach((d, i) => {
      const o = document.createElement("option");
      o.value = d.deviceId; o.textContent = d.label || `${kind === "audioinput" ? "Microphone" : "Speaker"} ${i + 1}`;
      sel.appendChild(o);
    });
    sel.value = keep;
  }
  if (!("setSinkId" in AudioContext.prototype)) {
    $("out-select").disabled = true;
    $("out-select").title = "This browser cannot choose the speaker; the system default is used.";
  }
}

// ---------------------------------------------------------------- playback (English speech)
class Player {
  constructor() { this.ctx = null; this.next = 0; this.sources = new Set(); }
  async open(sinkId) {
    this.ctx = new AudioContext();
    if (sinkId && this.ctx.setSinkId) { try { await this.ctx.setSinkId(sinkId); } catch (e) { log("could not use that speaker: " + e.message); } }
    this.next = 0;
  }
  play(int16, sampleRate) {
    const f = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) f[i] = int16[i] / 32768;
    const buf = this.ctx.createBuffer(1, f.length, sampleRate);
    buf.copyToChannel(f, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf; src.connect(this.ctx.destination);
    const at = Math.max(this.ctx.currentTime + 0.02, this.next);
    src.start(at);
    this.next = at + buf.duration + 0.15;      // short pause between sentences
    this.sources.add(src);
    src.onended = () => this.sources.delete(src);
  }
  beep(n = 1) {
    if (!this.ctx) return 0;
    let t = Math.max(this.ctx.currentTime + 0.02, this.next);
    for (let i = 0; i < n; i++) {
      const o = this.ctx.createOscillator(), g = this.ctx.createGain();
      o.frequency.value = 880; g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(0.25, t + 0.02); g.gain.linearRampToValueAtTime(0, t + 0.15);
      o.connect(g).connect(this.ctx.destination); o.start(t); o.stop(t + 0.16);
      t += 0.25;
    }
    this.next = t;
    return t - this.ctx.currentTime;
  }
  speaking(tail = 0.3) { return !!this.ctx && this.ctx.currentTime < this.next + tail; }
  stop() { for (const s of this.sources) { try { s.stop(); } catch (e) {} } this.sources.clear(); if (this.ctx) this.next = this.ctx.currentTime; }
  async close() { this.stop(); if (this.ctx) await this.ctx.close(); this.ctx = null; }
}

// ---------------------------------------------------------------- live session
let live = null;

async function startLive() {
  if (!$("consent-ok").checked) { setTurn("Please confirm that everyone being recorded has agreed.", "error"); return; }
  $("start-btn").disabled = true; $("upload-btn").disabled = true;
  clearUnits();
  const s = { ws: null, stream: null, cap: null, node: null, player: new Player(), waiting: false, ready: false,
              pendingAudio: null, units: new Map(), mode: $("mode-select").value, tts: $("speak").checked,
              headphones: $("headphones").checked };
  live = s;
  try {
    const micId = $("mic-select").value;
    s.stream = await navigator.mediaDevices.getUserMedia({ audio: {
      deviceId: micId ? { exact: micId } : undefined, channelCount: 1,
      echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    listDevices();
    await s.player.open($("out-select").value);
    s.cap = new AudioContext();
    await s.cap.audioWorklet.addModule("/static/capture-worklet.js");
    const srcNode = s.cap.createMediaStreamSource(s.stream);
    s.node = new AudioWorkletNode(s.cap, "capture-processor");
    srcNode.connect(s.node);
  } catch (e) {
    setTurn("Microphone unavailable: " + e.message, "error"); log("mic: " + e.message);
    await endLive(); return;
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  s.ws = new WebSocket(withToken(`${proto}://${location.host}/ws/live`));
  s.ws.binaryType = "arraybuffer";
  s.ws.onopen = () => {
    s.ws.send(JSON.stringify({ type: "start", sample_rate: s.cap.sampleRate, mode: s.mode, tts: s.tts,
                               partials: false, export: $("save").checked }));
    setTurn("Preparing…", "busy");
  };
  s.ws.onmessage = (m) => onMessage(s, m);
  s.ws.onclose = (e) => { if (e.code !== 1000 && e.reason) { setTurn(e.reason, "error"); log(`closed ${e.code}: ${e.reason}`); } endLive(); };
  s.ws.onerror = () => log("connection error");
  s.node.port.onmessage = (ev) => sendAudio(s, ev.data);
  $("stop-btn").disabled = false; $("cancel-btn").disabled = false;
}

function sendAudio(s, block) {
  // level meter
  let sum = 0; for (let i = 0; i < block.length; i++) sum += block[i] * block[i];
  const rms = Math.sqrt(sum / block.length);
  if (!s.ready || !s.ws || s.ws.readyState !== WebSocket.OPEN) return;
  // Echo guard: while English plays from the speaker the mic would hear it. Send silence instead.
  const muted = s.waiting || (s.tts && !s.headphones && s.player.speaking());
  $("level").style.width = muted ? "0%" : Math.min(100, rms * 400) + "%";
  if (muted) block = new Float32Array(block.length);
  s.ws.send(block.buffer);
}

function onMessage(s, m) {
  if (typeof m.data !== "string") {                 // binary: English speech for the preceding header
    if (s.pendingAudio) { s.player.play(new Int16Array(m.data), s.pendingAudio.sample_rate); s.pendingAudio = null; }
    return;
  }
  const msg = JSON.parse(m.data);
  switch (msg.type) {
    case "loading": setTurn("Loading models… (first start takes ~20 s)", "busy"); break;
    case "ready":
      s.ready = true;
      log(`session ${msg.session_id}: ${msg.models.asr} → ${msg.models.mt} → ${msg.models.tts || "no voice"}`);
      s.player.beep(1);
      setTurn(s.mode === "consecutive" ? "Speak Swahili now — one sentence, then pause" : "Speak Swahili now", "speak");
      break;
    case "tts_audio": s.pendingAudio = msg; break;
    case "event": onEvent(s, msg.event); break;
    case "turn":
      if (msg.state === "interpreting") { s.waiting = true; setTurn("Interpreting… (mic closed)", "busy"); }
      else if (msg.state === "spoken") waitThenListen(s);
      else if (msg.state === "cancelled") { s.waiting = false; setTurn("Cancelled — speak again", "speak"); }
      break;
    case "status": break;
    case "summary": onSummary(msg); break;
    case "error": log(`${msg.code}: ${msg.message}`); if (msg.code !== "bad_audio") setTurn(msg.message, "error"); break;
  }
}

function waitThenListen(s) {
  const tick = () => {
    if (live !== s) return;
    if (s.player.speaking(0.3)) { setTurn("Listening to the English… (mic closed)", "busy"); setTimeout(tick, 80); return; }
    const wait = s.player.beep(1);
    setTimeout(() => {
      if (live !== s) return;
      s.waiting = false;
      if (s.ws.readyState === WebSocket.OPEN) s.ws.send(JSON.stringify({ type: "listening" }));
      setTurn("Your turn — speak the next sentence", "speak");
    }, wait * 1000 + 50);
  };
  tick();
}

function onEvent(s, ev) {
  const d = ev.data || {};
  switch (ev.type) {
    case "vad_speech_start": $("hearing").textContent = "(hearing speech…)"; break;
    case "vad_speech_end": $("hearing").textContent = "(pause detected — interpreting…)"; break;
    case "asr_partial": $("hearing").textContent = "… " + (ev.text || ""); break;
    case "asr_final": $("hearing").textContent = ev.text ? "Heard: " + ev.text + (d.uncertain ? "  (uncertain)" : "") : ""; break;
    case "mt_final": addUnit(d.unit_id, d.source, ev.text, d.flags || []); break;
    case "pipeline_complete": addLatency(d); break;
    case "fallback": log(`fallback ${d.stage}: ${d.requested || d.from} → ${d.selected || d.to} (${d.reason})`); break;
    case "error": log(`error [${d.stage}]: ${d.error}`); break;
    case "status": if (d.status === "audio_overflow") log("pipeline full — audio dropped"); break;
  }
}

function clearUnits() { $("units").innerHTML = ""; $("empty").hidden = false; $("downloads").innerHTML = ""; $("latency-summary").textContent = ""; }

function addUnit(id, sw, en, flags) {
  $("empty").hidden = true;
  let li = document.querySelector(`#units li[data-unit="${id}"]`);
  if (!li) { li = document.createElement("li"); li.dataset.unit = id; $("units").appendChild(li); }
  li.innerHTML = `<div class="sw"></div><div class="en"></div><div class="lat"></div>`;
  li.querySelector(".sw").textContent = sw || "";
  li.querySelector(".en").textContent = en || "∅";
  for (const f of flags) li.querySelector(".lat").insertAdjacentHTML("beforeend", `<span class="chip flag">${escapeHtml(f)}</span>`);
  li.scrollIntoView({ block: "nearest" });
}

function addLatency(d) {
  const li = document.querySelector(`#units li[data-unit="${d.unit_id}"]`);
  if (!li) return;
  const lat = li.querySelector(".lat");
  const chips = [["Swahili text", d.lat_asr_final_ms], ["English text", d.lat_mt_final_ms], ["English voice", d.lat_first_audio_ms]];
  for (const [k, v] of chips) if (v != null) lat.insertAdjacentHTML("beforeend", `<span class="chip">${k} ${fmtS(v)}</span>`);
  if (d.tts_error) lat.insertAdjacentHTML("beforeend", `<span class="chip flag">voice failed</span>`);
}

function onSummary(msg) {
  const lat = msg.summary.latency_from_speech_end_ms || {};
  const fa = lat.first_audio || lat.mt_final;
  if (fa) $("latency-summary").textContent = `after you stop speaking: median ${fmtS(fa.median)}, p95 ${fmtS(fa.p95)} (${fa.n} sentences)`;
  showDownloads(msg.run_id, msg.files);
  log(`session ended (${(msg.summary.capture || {}).end || "stop"}); ${msg.summary.units} sentences`);
}

function showDownloads(runId, files) {
  const el = $("downloads");
  el.innerHTML = "";
  if (!runId) return;
  const links = [["export.zip", "Download all (zip)"], ["translation_en.txt", "English text"],
                 ["transcript_sw.txt", "Swahili text"], ["speech_en.wav", "English audio"]];
  for (const [f, label] of links) {
    if (f !== "export.zip" && !(files || []).includes(f)) continue;
    const a = document.createElement("a");
    a.href = withToken(`/api/runs/${encodeURIComponent(runId)}/${f}`); a.textContent = label; a.download = "";
    el.appendChild(a);
  }
}

function stopLive() {
  if (!live || !live.ws) return;
  $("stop-btn").disabled = true; $("cancel-btn").disabled = true;
  live.ready = false;                      // stop sending audio
  setTurn("Finishing what was heard…", "busy");
  if (live.ws.readyState === WebSocket.OPEN) live.ws.send(JSON.stringify({ type: "stop" }));
}

function cancelLive() {
  if (!live || !live.ws || live.ws.readyState !== WebSocket.OPEN) return;
  live.player.stop();
  live.ws.send(JSON.stringify({ type: "cancel" }));
}

async function endLive() {
  const s = live;
  if (!s) return;
  live = null;
  try { s.node && s.node.disconnect(); } catch (e) {}
  if (s.stream) s.stream.getTracks().forEach((t) => t.stop());
  if (s.cap) await s.cap.close().catch(() => {});
  // let the last English finish before closing the speaker
  setTimeout(() => s.player.close(), Math.max(0, (s.player.ctx ? (s.player.next - s.player.ctx.currentTime) : 0) * 1000) + 300);
  $("level").style.width = "0%";
  $("stop-btn").disabled = true; $("cancel-btn").disabled = true;
  if (!$("turn").classList.contains("error")) setTurn("Stopped", "idle");
  refreshStatus();
}

// ---------------------------------------------------------------- file upload
async function uploadFile(e) {
  e.preventDefault();
  const f = $("upload-file").files[0];
  if (!f) { $("upload-status").textContent = "Choose an audio file first."; return; }
  const fd = new FormData();
  fd.append("file", f); fd.append("tts", $("upload-tts").checked ? "true" : "false");
  $("upload-btn").disabled = true; $("start-btn").disabled = true;
  $("upload-status").textContent = `Interpreting ${f.name}… (about half the audio length on this computer)`;
  clearUnits();
  try {
    const r = await fetch("/api/interpret", { method: "POST", body: fd, headers: authHeaders() });
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || r.statusText);
    for (const u of body.units) { addUnit(u.unit_id, u.swahili, u.english, u.flags || []); }
    $("upload-status").textContent = `Done: ${body.units.length} sentences, ${body.summary.audio_in_s.toFixed(0)} s of audio in ${body.summary.wall_s.toFixed(0)} s.`;
    showDownloads(body.run_id, body.files);
    if (body.files.includes("speech_en.wav")) {
      const audio = document.createElement("audio");
      audio.controls = true; audio.src = withToken(`/api/runs/${encodeURIComponent(body.run_id)}/speech_en.wav`);
      $("downloads").appendChild(audio);
    }
  } catch (err) {
    $("upload-status").textContent = "Failed: " + err.message;
  } finally { refreshStatus(); }
}

// ---------------------------------------------------------------- wiring
$("start-btn").onclick = startLive;
$("stop-btn").onclick = stopLive;
$("cancel-btn").onclick = cancelLive;
$("upload-form").onsubmit = uploadFile;
$("mode-select").onchange = () => { if ($("mode-select").value === "simultaneous" && !$("headphones").checked) log("tip: continuous mode works best with headphones"); };
listDevices();
refreshStatus();
setInterval(refreshStatus, 3000);
