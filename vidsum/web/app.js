"use strict";

/* vidsum 本地控制台前端：读表单 -> POST /api/run，轮询 /api/status 渲染实时产物。
   每个面板都先做内容比对再更新 DOM，轮询期间画面不会无谓跳动。 */

const $ = (id) => document.getElementById(id);

const STATE_LABELS = {
  idle: "空闲",
  running: "运行中",
  done: "已完成",
  error: "出错",
  stopped: "已停止",
};
const EMPTY_TEXT = {
  transcript: "暂无转写稿",
  summary: "暂无摘要",
  final: "暂无终稿",
};
const RECENT_KEY = "vidsum-recent";
const SETTINGS_KEY = "vidsum-settings";
const POLL_MS = 1500;

let pollTimer = null;
let lastState = "";
let lastDry = false;
let toastTimer = null;
let saveTimer = null;

/* ---------- 小工具 ---------- */

function esc(text) {
  return String(text == null ? "" : text).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fmtClock(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

function toast(message, kind) {
  const el = $("toast");
  el.textContent = message;
  el.className = "toast" + (kind ? " " + kind : "");
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
}

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  let data = {};
  try { data = await res.json(); } catch (e) { data = {}; }
  if (!res.ok && !data.error) data.error = `请求失败（HTTP ${res.status}）`;
  return data;
}

/* ---------- Markdown 轻量渲染 ---------- */

const TS_LINE = /^\*\*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\*\*\s*(.*)$/;

function inline(text) {
  return text
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
}

function renderMarkdown(source) {
  const out = [];
  let para = [];
  let list = [];
  let quote = [];

  const flushPara = () => {
    if (para.length) { out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; }
  };
  const flushList = () => {
    if (list.length) {
      out.push(`<ul>${list.map((t) => `<li>${inline(t)}</li>`).join("")}</ul>`);
      list = [];
    }
  };
  const flushQuote = () => {
    if (quote.length) {
      out.push(`<blockquote>${quote.map(inline).join("<br>")}</blockquote>`);
      quote = [];
    }
  };
  const flushAll = () => { flushPara(); flushList(); flushQuote(); };

  // 结构标记在转义前匹配：">" 若先变成 "&gt;" 就再也认不出引用块
  for (const raw of String(source || "").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) { flushAll(); continue; }

    const ts = line.match(TS_LINE);
    if (ts) {
      flushAll();
      out.push(`<div class="ts-row"><span class="ts">${ts[1]}` +
               `</span><span class="ts-text">${inline(esc(ts[2]))}</span></div>`);
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      flushAll();
      const level = heading[1].length;
      out.push(`<h${level}>${inline(esc(heading[2]))}</h${level}>`);
      continue;
    }

    const image = line.match(/^!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)$/);
    if (image) {
      flushAll();
      // 终稿里的帧图是工作目录相对路径，网页上经 /work/ 路由才能取到
      let src = image[2];
      if (!/^(?:https?:)?\/\//i.test(src) && !src.startsWith("/")) {
        src = "/work/" + src;
      }
      out.push(`<figure><img src="${esc(src)}" alt="${esc(image[1])}" loading="lazy"></figure>`);
      continue;
    }

    if (/^(?:-{3,}|\*{3,}|_{3,})$/.test(line)) { flushAll(); out.push("<hr>"); continue; }

    const quoted = line.match(/^>\s?(.*)$/);
    if (quoted) { flushPara(); flushList(); quote.push(esc(quoted[1])); continue; }

    const item = line.match(/^(?:[-*+]|\d+[.)])\s+(.*)$/);
    if (item) { flushPara(); flushQuote(); list.push(esc(item[1])); continue; }

    flushList(); flushQuote();
    para.push(esc(line));
  }
  flushAll();
  return out.join("\n");
}

/* ---------- 表单 ---------- */

function readForm() {
  const num = (id, fallback) => {
    const n = parseFloat($(id).value);
    return Number.isFinite(n) ? n : fallback;
  };
  const params = {
    input: $("input").value.trim(),
    asr: currentTrack() === "cloud" ? "api" : $("asr-local").value,
    audio_backend: $("audio-backend").value,
    language: $("language").value.trim() || "zh",
    chunk_minutes: num("chunk-minutes", 1),
    workers: Math.max(1, Math.round(num("workers", 2))),
    model_size: $("model-size").value,
    sherpa_threads: Math.max(1, Math.round(num("sherpa-threads", 2))),
    frames_per_segment: Math.max(0, Math.round(num("frames-per-segment", 2))),
    frame_width: Math.max(160, Math.round(num("frame-width", 480))),
    no_punctuation: $("no-punctuation").checked,
    no_frames: $("no-frames").checked,
    vision: $("vision").checked,
    no_summary: $("no-summary").checked,
    keep_audio: $("keep-audio").checked,
    keep_chunks: $("keep-chunks").checked,
    no_silence_split: $("no-silence-split").checked,
    force: $("force").checked,
  };
  const modelDir = $("sherpa-model-dir").value.trim();
  if (modelDir) params.sherpa_model_dir = modelDir;
  const whisperDir = $("whisper-download-root").value.trim();
  if (whisperDir) params.whisper_download_root = whisperDir;
  const output = $("output").value.trim();
  if (output) params.output = output;
  // 云端字段始终读取：在本地/云端两条轨道间切换时不丢已填的接口配置
  params.asr_base_url = $("asr-base-url").value.trim();
  params.asr_model = $("asr-model").value.trim();
  params.asr_api_key = $("asr-api-key").value;
  if (!params.no_summary) {
    params.llm_model = $("llm-model").value.trim();
    params.llm_base_url = $("llm-base-url").value.trim();
    params.llm_api_key = $("llm-api-key").value;
    params.llm_max_tokens = Math.max(256, Math.round(num("llm-max-tokens", 2048)));
  }
  const ua = $("user-agent").value.trim();
  if (ua) params.user_agent = ua;
  const referer = $("referer").value.trim();
  if (referer) params.referer = referer;
  const cookie = $("cookie").value;
  if (cookie) params.cookie = cookie;
  return params;
}

function isNetworkInput(value) {
  return /^https?:\/\//i.test(String(value || "").trim());
}

function currentTrack() {
  const active = document.querySelector("#asr-track .seg.active");
  return active ? active.dataset.track : "local";
}

function setTrack(track) {
  document.querySelectorAll("#asr-track .seg").forEach((b) => {
    b.classList.toggle("active", b.dataset.track === track);
  });
}

function syncVisibility() {
  const track = currentTrack();
  $("track-local").hidden = track !== "local";
  $("track-cloud").hidden = track !== "cloud";
  const local = $("asr-local").value;
  $("wrap-model-size").hidden = track !== "local" || local !== "faster-whisper";
  $("wrap-whisper-dir").hidden = track !== "local" || local !== "faster-whisper";
  $("wrap-sherpa").hidden = track !== "local" || local !== "sherpa";
  $("wrap-llm").hidden = $("no-summary").checked;
  $("wrap-network").hidden = !isNetworkInput($("input").value);
}

function loadJson(key, fallback) {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || "null");
    return parsed == null ? fallback : parsed;
  } catch (e) {
    return fallback;
  }
}

function saveSettings() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(readForm())); } catch (e) {}
  }, 250);
}

function restoreSettings() {
  const s = loadJson(SETTINGS_KEY, {});
  const put = (id, value) => {
    const el = $(id);
    if (!el || value == null) return;
    if (el.tagName === "SELECT") {
      const ok = Array.from(el.options).some((o) => o.value === value || o.text === value);
      if (ok) el.value = value;
      return;
    }
    if (el.type === "checkbox") el.checked = !!value;
    else el.value = value;
  };
  put("input", s.input);
  setTrack(s.asr === "api" ? "cloud" : "local");
  put("asr-local", s.asr === "api" ? "sherpa" : s.asr);
  put("model-size", s.model_size);
  put("audio-backend", s.audio_backend);
  put("language", s.language);
  put("chunk-minutes", s.chunk_minutes);
  put("workers", s.workers);
  put("sherpa-model-dir", s.sherpa_model_dir);
  put("whisper-download-root", s.whisper_download_root);
  put("sherpa-threads", s.sherpa_threads);
  put("frames-per-segment", s.frames_per_segment);
  put("frame-width", s.frame_width);
  put("no-frames", s.no_frames);
  put("vision", s.vision);
  put("llm-model", s.llm_model);
  put("llm-base-url", s.llm_base_url);
  put("llm-api-key", s.llm_api_key);
  put("llm-max-tokens", s.llm_max_tokens);
  put("output", s.output);
  put("asr-base-url", s.asr_base_url);
  put("asr-model", s.asr_model);
  put("asr-api-key", s.asr_api_key);
  put("user-agent", s.user_agent);
  put("referer", s.referer);
  put("cookie", s.cookie);
  put("no-punctuation", s.no_punctuation);
  put("no-summary", s.no_summary);
  put("keep-audio", s.keep_audio);
  put("keep-chunks", s.keep_chunks);
  put("no-silence-split", s.no_silence_split);
  put("force", s.force);
  syncVisibility();
}

function renderRecent() {
  const list = loadJson(RECENT_KEY, []);
  $("recent").innerHTML = list.map((x) => `<option value="${esc(x)}"></option>`).join("");
}

function rememberInput(value) {
  if (!value) return;
  const list = [value, ...loadJson(RECENT_KEY, []).filter((x) => x !== value)].slice(0, 8);
  try { localStorage.setItem(RECENT_KEY, JSON.stringify(list)); } catch (e) {}
  renderRecent();
}

/* ---------- 动作 ---------- */

async function startRun(dry) {
  const params = readForm();
  if (!params.input) {
    toast("请先填写输入路径或直链", "error");
    $("input").focus();
    return;
  }
  if (dry) params.dry_run = true;
  lastDry = !!dry;
  saveSettings();
  let res;
  try { res = await post("/api/run", params); }
  catch (e) { toast(`无法连接服务：${e.message}`, "error"); return; }
  if (!res.ok) { toast(res.error || "启动失败", "error"); return; }
  rememberInput(params.input);
  toast(dry ? "试运行已启动" : "任务已启动", "ok");
  switchTab("logs");
  poll();
}

async function stopRun() {
  let res;
  try { res = await post("/api/stop", {}); }
  catch (e) { toast(`无法连接服务：${e.message}`, "error"); return; }
  if (res.ok) toast("已请求停止", "ok");
  else toast(res.error || "停止失败", "error");
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === name);
  });
  document.querySelectorAll(".pane").forEach((p) => {
    p.hidden = p.dataset.pane !== name;
  });
}

function syncThemeIcon() {
  const dark = document.documentElement.dataset.theme === "dark";
  document.querySelector("#theme-toggle use")
    .setAttribute("href", dark ? "#i-moon" : "#i-sun");
}

/* ---------- 状态渲染 ---------- */

function renderDoc(el, text, icon, emptyText) {
  const source = text || "";
  if (el._source === source) return;
  el._source = source;
  const pane = el.closest(".pane");
  const pinned = pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 80;
  if (!source.trim()) {
    el.innerHTML = `<div class="empty"><svg class="icon"><use href="#${icon}"/>` +
                   `</svg><span>${emptyText}</span></div>`;
  } else {
    el.innerHTML = renderMarkdown(source);
  }
  if (pinned) pane.scrollTop = pane.scrollHeight;
}

function renderLogs(lines) {
  const el = $("out-logs");
  const text = (lines || []).join("\n");
  if (el._text === text) return;
  const pane = el.closest(".pane");
  const pinned = pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 80;
  el.textContent = text;
  el._text = text;
  if (pinned) pane.scrollTop = pane.scrollHeight;
}

function renderFiles(files) {
  const el = $("out-files");
  const list = files || [];
  const sig = list.map((f) => `${f.name}:${f.size}`).join("|");
  if (el._sig === sig) return;
  el._sig = sig;
  if (!list.length) {
    el.innerHTML = '<tr><td colspan="2" class="muted">（暂无文件）</td></tr>';
    return;
  }
  el.innerHTML = list.map((f) =>
    `<tr><td>${esc(f.name)}</td><td class="num">${esc(f.pretty || f.size)}</td></tr>`
  ).join("");
}

function renderStatus(s) {
  const state = s.state || "idle";
  const running = state === "running";

  const pill = $("state-pill");
  pill.className = `pill pill-${state}`;
  $("state-text").textContent = STATE_LABELS[state] || state;

  const progress = s.progress || {};
  const total = Number(progress.total) || 0;
  const done = Number(progress.done) || 0;
  const pct = total > 0
    ? Math.min(100, Math.round((done / total) * 100))
    : (state === "done" ? 100 : 0);
  $("progress-bar").style.width = `${pct}%`;
  $("progress-text").textContent = total > 0 ? `${done}/${total}` : "–";
  $("progress-text").title = `已并入摘要 ${Number(progress.summarized) || 0} 段`;

  $("elapsed").textContent = fmtClock(s.elapsed || 0);
  const work = s.work || "";
  const workEl = $("work-dir");
  workEl.textContent = work;
  workEl.title = work;

  $("btn-run").disabled = running;
  $("btn-dry").disabled = running;
  $("btn-stop").disabled = !running;

  renderDoc($("out-transcript"), s.transcript, "i-file", EMPTY_TEXT.transcript);
  renderDoc($("out-summary"), s.summary, "i-list", EMPTY_TEXT.summary);
  renderDoc($("out-final"), s.final, "i-star", EMPTY_TEXT.final);
  renderLogs(s.logs);
  renderFiles(s.files);

  if (state !== lastState) {
    if (state === "done") toast(lastDry ? "试运行完成" : "处理完成", "ok");
    else if (state === "error") toast(s.error || "处理出错", "error");
    else if (state === "stopped") toast("已停止");
    lastState = state;
  }
}

async function poll() {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    renderStatus(await res.json());
  } catch (e) { /* 服务尚未就绪时保持现状 */ }
}

/* ---------- 启动 ---------- */

function init() {
  restoreSettings();
  renderRecent();
  syncThemeIcon();
  switchTab("transcript");

  $("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("vidsum-theme", next); } catch (e) {}
    syncThemeIcon();
  });

  $("asr-track").addEventListener("click", (e) => {
    const seg = e.target.closest(".seg");
    if (!seg) return;
    setTrack(seg.dataset.track);
    syncVisibility();
    saveSettings();
  });
  $("asr-local").addEventListener("change", syncVisibility);
  $("input").addEventListener("input", syncVisibility);
  $("no-summary").addEventListener("change", syncVisibility);
  $("form").addEventListener("change", saveSettings);
  $("form").addEventListener("input", saveSettings);
  $("form").addEventListener("submit", (e) => { e.preventDefault(); startRun(false); });

  $("btn-run").addEventListener("click", () => startRun(false));
  $("btn-dry").addEventListener("click", () => startRun(true));
  $("btn-stop").addEventListener("click", stopRun);

  $("tabs").addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (tab) switchTab(tab.dataset.tab);
  });

  poll();
  pollTimer = setInterval(poll, POLL_MS);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
