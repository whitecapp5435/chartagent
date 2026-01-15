import argparse
import html
import json
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 0
    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if not isinstance(obj, dict):
                    obj = {"_raw": s, "error": "Line is not a JSON object"}
            except Exception as e:
                obj = {"_raw": s, "error": f"JSON parse error: {e}"}
            obj["__idx"] = idx
            obj["__line"] = line_no
            idx += 1
            rows.append(obj)
    return rows


def _safe_relpath(p: str) -> str:
    s = str(p or "")
    s = s.replace("\\", "/")
    s = s.lstrip("/")
    s = os.path.normpath(s).replace("\\", "/")
    # normpath can return "." for empty input
    if s == ".":
        s = ""
    return s


@dataclass(frozen=True)
class DirEntry:
    name: str
    rel_path: str
    is_dir: bool
    size: int
    mtime_iso: str


class PredictionsWebApp:
    def __init__(self, *, predictions_path: str, base_dir: str) -> None:
        self.predictions_path = Path(predictions_path).resolve()
        self.base_dir = Path(base_dir).resolve()
        self.rows = _iter_jsonl(self.predictions_path)

    def _resolve(self, rel_path: str) -> Path:
        rel = _safe_relpath(rel_path)
        if not rel:
            return self.base_dir
        p = (self.base_dir / rel).resolve()
        # Ensure within base_dir.
        if p != self.base_dir and self.base_dir not in p.parents:
            raise PermissionError(f"path escapes base_dir: {rel_path!r}")
        return p

    def stat(self, rel_path: str) -> Optional[Dict[str, Any]]:
        try:
            p = self._resolve(rel_path)
        except Exception:
            return None
        if not p.exists():
            return None
        st = p.stat()
        return {
            "rel_path": _safe_relpath(rel_path),
            "is_dir": p.is_dir(),
            "size": int(st.st_size),
            "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(),
        }

    def listdir(self, rel_dir: str) -> List[DirEntry]:
        p = self._resolve(rel_dir)
        if not p.exists():
            raise FileNotFoundError(rel_dir)
        if not p.is_dir():
            raise NotADirectoryError(rel_dir)
        out: List[DirEntry] = []
        for child in p.iterdir():
            try:
                st = child.stat()
            except Exception:
                continue
            rel = child.relative_to(self.base_dir).as_posix()
            out.append(
                DirEntry(
                    name=child.name,
                    rel_path=rel,
                    is_dir=child.is_dir(),
                    size=int(st.st_size),
                    mtime_iso=datetime.fromtimestamp(st.st_mtime).isoformat(),
                )
            )
        out.sort(key=lambda e: (not e.is_dir, e.name.lower()))
        return out

    def read_file(self, rel_path: str) -> Tuple[bytes, str]:
        p = self._resolve(rel_path)
        if not p.exists():
            raise FileNotFoundError(rel_path)
        if p.is_dir():
            raise IsADirectoryError(rel_path)

        ext = p.suffix.lower()
        ctype, _enc = mimetypes.guess_type(p.name)
        if ext in {".txt", ".md", ".log", ".out", ".py", ".jsonl"}:
            ctype = "text/plain; charset=utf-8"
        if ext in {".json"}:
            ctype = "application/json; charset=utf-8"
        if not ctype:
            ctype = "application/octet-stream"
        return p.read_bytes(), ctype


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Predictions Viewer</title>
  <style>
    :root{
      --bg:#0b0f14; --panel:#0f1723; --muted:#9fb0c3; --fg:#e6edf3;
      --line:#223044; --accent:#4aa3ff; --bad:#ff5c5c; --good:#35d07f;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif;
    }
    *{box-sizing:border-box}
    body{margin:0; font-family:var(--sans); background:var(--bg); color:var(--fg)}
    header{padding:12px 14px; border-bottom:1px solid var(--line); background:linear-gradient(180deg,#0f1723, #0b0f14)}
    header h1{margin:0 0 6px; font-size:16px}
    .hint{margin-top:6px; font-size:12px; color:var(--muted); line-height:1.4}
    .status{margin-top:6px; font-size:12px; color:var(--muted)}
    .layout{display:flex; height:calc(100vh - 110px)}
    .left{width:44%; min-width:360px; border-right:1px solid var(--line); display:flex; flex-direction:column}
    .right{flex:1; display:flex; flex-direction:column}
    .controls{padding:10px 14px; border-bottom:1px solid var(--line); background:rgba(15,23,35,0.45)}
    .row{display:flex; gap:10px; flex-wrap:wrap; align-items:center}
    .field{display:flex; flex-direction:column; gap:6px; min-width:180px}
    .field label{font-size:12px; color:var(--muted)}
    input[type="text"], select{
      background:var(--panel); border:1px solid var(--line); color:var(--fg);
      padding:8px; border-radius:8px; outline:none;
    }
    .tableWrap{flex:1; overflow:auto}
    table{width:100%; border-collapse:collapse; font-size:12px}
    thead th{position:sticky; top:0; background:rgba(15,23,35,0.95); border-bottom:1px solid var(--line); text-align:left; padding:10px 10px}
    tbody td{border-bottom:1px solid rgba(34,48,68,0.6); padding:8px 10px; vertical-align:top}
    tbody tr:hover{background:rgba(74,163,255,0.08); cursor:pointer}
    tbody tr.selected{background:rgba(74,163,255,0.16)}
    .pill{display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid var(--line); color:var(--muted); font-size:11px}
    .pill.bad{border-color:rgba(255,92,92,0.5); color:var(--bad)}
    .pill.good{border-color:rgba(53,208,127,0.5); color:var(--good)}
    .detail{flex:1; overflow:auto; padding:14px}
    .empty{color:var(--muted); font-size:13px; padding:14px}
    .grid{display:grid; grid-template-columns: 360px 1fr; gap:12px}
    .card{background:rgba(15,23,35,0.6); border:1px solid var(--line); border-radius:12px; padding:12px}
    .card h2{margin:0 0 8px; font-size:13px; color:var(--muted); font-weight:600}
    .kv{display:grid; grid-template-columns: 140px 1fr; gap:6px 10px; font-size:12px}
    .kv .k{color:var(--muted)}
    .mono{font-family:var(--mono)}
    img.preview{max-width:100%; border-radius:10px; border:1px solid var(--line); background:#000}
    .stepsBar{display:flex; flex-wrap:wrap; gap:6px; margin:12px 0}
    .stepBtn{
      border:1px solid var(--line); background:rgba(15,23,35,0.6); color:var(--fg);
      padding:6px 10px; border-radius:10px; font-size:12px; cursor:pointer;
    }
    .stepBtn.active{border-color:rgba(74,163,255,0.9); box-shadow:0 0 0 2px rgba(74,163,255,0.15) inset}
    .cols{display:grid; grid-template-columns:1fr 1fr; gap:12px}
    pre{
      margin:0; padding:10px; border-radius:10px; background:rgba(11,15,20,0.8);
      border:1px solid rgba(34,48,68,0.8); overflow:auto; max-height:320px;
      font-family:var(--mono); font-size:11px; line-height:1.35;
      white-space:pre-wrap;
    }
    .files{display:grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap:10px}
    .fileItem{border:1px solid var(--line); background:rgba(15,23,35,0.45); border-radius:12px; padding:10px}
    .fileItem .path{font-family:var(--mono); font-size:11px; color:var(--muted); word-break:break-all}
    .fileItem button{
      margin-top:8px; border:1px solid var(--line); background:rgba(74,163,255,0.12); color:var(--fg);
      padding:6px 10px; border-radius:10px; cursor:pointer; font-size:12px;
    }
    .modal{position:fixed; inset:0; background:rgba(0,0,0,0.65); display:none; align-items:center; justify-content:center; padding:20px}
    .modal.show{display:flex}
    .modalInner{width:min(1100px, 100%); max-height:92vh; overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px}
    .modalTop{display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px}
    .modalTop .title{font-family:var(--mono); font-size:12px; color:var(--muted); word-break:break-all}
    .modalTop button{border:1px solid var(--line); background:rgba(255,255,255,0.06); color:var(--fg); padding:6px 10px; border-radius:10px; cursor:pointer}
    .modalBody img{max-width:100%; border-radius:12px; border:1px solid var(--line)}
    @media (max-width: 1100px){
      .layout{flex-direction:column; height:auto}
      .left{width:auto; min-width:unset; height:52vh}
      .right{height:48vh}
      .grid{grid-template-columns: 1fr}
      .cols{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
<header>
  <h1>Predictions / Run Viewer</h1>
  <div class="hint">
    왼쪽에서 row를 클릭하면 오른쪽에서 <span class="mono">run_dir/steps</span> 기반으로 step별 prompt/model/obs/artifacts를 렌더링합니다.
  </div>
  <div id="status" class="status">loading…</div>
</header>

<div class="layout">
  <div class="left">
    <div class="controls">
      <div class="row">
        <div class="field" style="flex:1; min-width:240px">
          <label>검색 (image_path / error / run_id)</label>
          <input id="q" type="text" placeholder="예: dual axis / 20251222_203743 / Max steps" />
        </div>
        <div class="field">
          <label>split</label>
          <select id="split">
            <option value="">all</option>
            <option value="dev">dev</option>
            <option value="val">val</option>
          </select>
        </div>
        <div class="field">
          <label>옵션</label>
          <div style="display:flex; gap:10px; align-items:center; padding:6px 0">
            <label style="display:flex; gap:6px; align-items:center; font-size:12px; color:var(--muted)">
              <input id="errOnly" type="checkbox" /> error만
            </label>
            <label style="display:flex; gap:6px; align-items:center; font-size:12px; color:var(--muted)">
              <input id="hasRun" type="checkbox" /> run_dir만
            </label>
            <label style="display:flex; gap:6px; align-items:center; font-size:12px; color:var(--muted)">
              <input id="dedupe" type="checkbox" /> image_path dedupe(마지막)
            </label>
          </div>
        </div>
      </div>
      <div class="status" id="counts"></div>
    </div>

    <div class="tableWrap">
      <table>
        <thead>
          <tr>
            <th style="width:64px">idx</th>
            <th style="width:64px">split</th>
            <th>image</th>
            <th style="width:120px">y_true</th>
            <th style="width:120px">y_pred</th>
            <th style="width:110px">status</th>
            <th style="width:160px">run_id</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>

  <div class="right">
    <div class="detail" id="detail">
      <div id="empty" class="empty">왼쪽에서 row를 클릭하세요.</div>

      <div id="content" style="display:none">
        <div class="grid">
          <div class="card">
            <h2>Image</h2>
            <img id="img" class="preview" alt="sample image" />
            <div class="hint" id="imgHint"></div>
          </div>

          <div class="card">
            <h2>Row</h2>
            <div class="kv">
              <div class="k">index</div><div class="mono" id="rowIdx"></div>
              <div class="k">split</div><div id="rowSplit"></div>
              <div class="k">image_path</div><div class="mono" id="rowImage"></div>
              <div class="k">run_dir</div><div class="mono" id="rowRun"></div>
              <div class="k">y_true</div><div class="mono" id="rowTrue"></div>
              <div class="k">y_pred</div><div class="mono" id="rowPred"></div>
              <div class="k">error</div><div class="mono" id="rowErr"></div>
            </div>
            <div style="margin-top:10px">
              <details>
                <summary style="cursor:pointer; color:var(--muted); font-size:12px">raw row JSON 보기</summary>
                <pre id="rowRaw"></pre>
              </details>
            </div>
          </div>
        </div>

        <div class="card" style="margin-top:12px">
          <h2>Run metadata</h2>
          <div class="cols">
            <div>
              <div class="hint mono">answer.json</div>
              <pre id="answerJson">(run_dir 선택 후 로드됨)</pre>
            </div>
            <div>
              <div class="hint mono">metadata.json / metadata_info.json / metadata_error.json</div>
              <pre id="metaJson">(run_dir 선택 후 로드됨)</pre>
            </div>
          </div>
        </div>

        <div class="card" style="margin-top:12px">
          <h2>Steps</h2>
          <div id="stepsBar" class="stepsBar"></div>
          <div class="cols">
            <div>
              <div class="hint mono" id="stepLabelPrompt">prompt.txt</div>
              <pre id="promptTxt"></pre>
            </div>
            <div>
              <div class="hint mono" id="stepLabelModel">model_output.txt</div>
              <pre id="modelTxt"></pre>
            </div>
          </div>
          <div class="cols" style="margin-top:12px">
            <div>
              <div class="hint mono">action.json</div>
              <pre id="actionJson"></pre>
            </div>
            <div>
              <div class="hint mono">observation.json</div>
              <pre id="obsJson"></pre>
            </div>
          </div>

          <div style="margin-top:12px">
            <div class="hint">Artifacts</div>
            <div id="files" class="files"></div>
          </div>
        </div>

        <div class="card" style="margin-top:12px">
          <h2>Trace (JSONL 내 trace/tool_sequence)</h2>
          <pre id="tracePre"></pre>
        </div>
      </div>
    </div>
  </div>
</div>

<div id="modal" class="modal" role="dialog" aria-modal="true">
  <div class="modalInner">
    <div class="modalTop">
      <div class="title" id="modalTitle"></div>
      <div style="display:flex; gap:8px">
        <a id="modalDownload" href="#" download style="text-decoration:none; color:var(--fg); border:1px solid var(--line); padding:6px 10px; border-radius:10px; background:rgba(255,255,255,0.06)">Download</a>
        <button id="modalClose">Close</button>
      </div>
    </div>
    <div class="modalBody" id="modalBody"></div>
  </div>
</div>

<script>
(() => {
  const $ = (id) => document.getElementById(id);
  const api = {
    rows: "/api/rows",
    listdir: (p) => "/api/listdir?path=" + encodeURIComponent(p),
    file: (p) => "/file?path=" + encodeURIComponent(p),
  };

  const state = {
    rows: [],
    rowById: new Map(),
    selectedId: null,
    selectedStep: null,
  };

  const els = {
    status: $("status"),
    counts: $("counts"),
    q: $("q"),
    split: $("split"),
    errOnly: $("errOnly"),
    hasRun: $("hasRun"),
    dedupe: $("dedupe"),
    tbody: $("tbody"),
    empty: $("empty"),
    content: $("content"),
    img: $("img"),
    imgHint: $("imgHint"),
    rowIdx: $("rowIdx"),
    rowSplit: $("rowSplit"),
    rowImage: $("rowImage"),
    rowRun: $("rowRun"),
    rowTrue: $("rowTrue"),
    rowPred: $("rowPred"),
    rowErr: $("rowErr"),
    rowRaw: $("rowRaw"),
    answerJson: $("answerJson"),
    metaJson: $("metaJson"),
    stepsBar: $("stepsBar"),
    promptTxt: $("promptTxt"),
    modelTxt: $("modelTxt"),
    actionJson: $("actionJson"),
    obsJson: $("obsJson"),
    files: $("files"),
    tracePre: $("tracePre"),
    stepLabelPrompt: $("stepLabelPrompt"),
    stepLabelModel: $("stepLabelModel"),
    modal: $("modal"),
    modalBody: $("modalBody"),
    modalTitle: $("modalTitle"),
    modalDownload: $("modalDownload"),
    modalClose: $("modalClose"),
  };

  function shortImageName(imagePath){
    const p = String(imagePath || "");
    const parts = p.replace(/\\\\/g,"/").split("/");
    return parts[parts.length - 1] || p;
  }

  function getRunId(runDir){
    const p = String(runDir || "").replace(/\\\\/g,"/").replace(/\\/+$/,"");
    if (!p) return "";
    const parts = p.split("/");
    return parts[parts.length - 1] || p;
  }

  function normalizeLabelList(x){
    if (Array.isArray(x)) return x.map(String);
    if (typeof x === "string") return [x];
    return [];
  }

  function dedupeByImagePath(rows){
    const m = new Map();
    for (const r of rows){
      const k = String(r.image_path || "");
      const key = k ? k : ("__no_image__:" + r.__idx);
      if (m.has(key)) m.delete(key);
      m.set(key, r);
    }
    return Array.from(m.values());
  }

  function filteredRows(){
    let rows = state.rows.slice();
    if (els.dedupe.checked) rows = dedupeByImagePath(rows);

    const q = String(els.q.value || "").trim().toLowerCase();
    const split = String(els.split.value || "").trim().toLowerCase();
    const errOnly = !!els.errOnly.checked;
    const hasRun = !!els.hasRun.checked;

    rows = rows.filter(r => {
      if (split && String(r.split || "").toLowerCase() !== split) return false;
      if (hasRun && !String(r.run_dir || "").trim()) return false;
      const hasErr = !!String(r.error || "").trim();
      if (errOnly && !hasErr) return false;
      if (q) {
        const hay = [
          String(r.image_path || ""),
          String(r.error || ""),
          String(r.run_dir || ""),
          String(r.split || ""),
          String((r.y_true || []).join(",")),
          String((r.y_pred || []).join(",")),
        ].join(" ").toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    return rows;
  }

  function renderTable(){
    const rows = filteredRows();
    els.tbody.innerHTML = "";
    for (const r of rows){
      const tr = document.createElement("tr");
      tr.dataset.id = String(r.__idx);
      if (state.selectedId !== null && r.__idx === state.selectedId) tr.classList.add("selected");

      const yTrue = normalizeLabelList(r.y_true).join(", ");
      const yPred = normalizeLabelList(r.y_pred).join(", ");
      const err = String(r.error || "").trim();
      const hasErr = !!err;
      const statusPill = hasErr
        ? `<span class="pill bad">error</span>`
        : `<span class="pill good">ok</span>`;

      tr.innerHTML = `
        <td class="mono">${r.__idx}</td>
        <td>${String(r.split || "")}</td>
        <td title="${String(r.image_path || "")}">${shortImageName(r.image_path)}</td>
        <td class="mono">${yTrue}</td>
        <td class="mono">${yPred}</td>
        <td>${statusPill}</td>
        <td class="mono" title="${String(r.run_dir || "")}">${getRunId(r.run_dir)}</td>
      `;
      els.tbody.appendChild(tr);
    }
    els.counts.textContent = `rows: ${rows.length} / ${els.dedupe.checked ? "deduped" : "all"} ${state.rows.length}`;
  }

  async function fetchTextMaybe(relPath){
    if (!relPath) return null;
    try{
      const resp = await fetch(api.file(relPath));
      if (!resp.ok) return null;
      return await resp.text();
    }catch{
      return null;
    }
  }

  async function fetchJsonMaybe(relPath){
    const t = await fetchTextMaybe(relPath);
    if (t === null) return null;
    try { return JSON.parse(t); }
    catch(e){ return { _parse_error: String(e), _raw: String(t).slice(0, 5000) }; }
  }

  function jsonPretty(x){
    try { return JSON.stringify(x, null, 2); }
    catch { return String(x); }
  }

  function setPre(el, text){
    el.textContent = text === null ? "(missing)" : String(text);
  }

  async function listDir(relDir){
    try{
      const resp = await fetch(api.listdir(relDir));
      if (!resp.ok) return [];
      const obj = await resp.json();
      if (obj && Array.isArray(obj.entries)) return obj.entries;
      return [];
    }catch{
      return [];
    }
  }

  function showModalForFile(relPath){
    els.modalTitle.textContent = relPath;
    els.modalBody.innerHTML = "";
    els.modalDownload.href = api.file(relPath);
    els.modalDownload.download = String(relPath).split("/").pop() || "download";

    const ext = (String(relPath).split(".").pop() || "").toLowerCase();
    const isImg = ["png","jpg","jpeg","gif","webp","bmp"].includes(ext);

    if (isImg) {
      const img = document.createElement("img");
      img.src = api.file(relPath);
      img.alt = relPath;
      els.modalBody.appendChild(img);
    } else {
      fetchTextMaybe(relPath).then(txt => {
        const pre = document.createElement("pre");
        const MAX = 220000;
        const s = String(txt || "");
        pre.textContent = s.length > MAX ? (s.slice(0, MAX) + "\\n\\n...(truncated)...") : s;
        els.modalBody.appendChild(pre);
      });
    }
    els.modal.classList.add("show");
  }

  function closeModal(){
    els.modal.classList.remove("show");
    els.modalBody.innerHTML = "";
  }
  els.modalClose.addEventListener("click", closeModal);
  els.modal.addEventListener("click", (e) => { if (e.target === els.modal) closeModal(); });

  async function loadAndShowRunMeta(runDir){
    if (!runDir) {
      els.answerJson.textContent = "(no run_dir)";
      els.metaJson.textContent = "(no run_dir)";
      return;
    }
    const answer = await fetchJsonMaybe(`${runDir}/answer.json`);
    const meta = await fetchJsonMaybe(`${runDir}/metadata.json`);
    const metaInfo = await fetchJsonMaybe(`${runDir}/metadata_info.json`);
    const metaErr = await fetchJsonMaybe(`${runDir}/metadata_error.json`);
    els.answerJson.textContent = answer ? jsonPretty(answer) : "(missing)";
    els.metaJson.textContent = jsonPretty({ metadata: meta, metadata_info: metaInfo, metadata_error: metaErr });
  }

  async function renderStep(runDir, stepId){
    state.selectedStep = stepId;
    for (const btn of els.stepsBar.querySelectorAll("button.stepBtn")){
      btn.classList.toggle("active", btn.dataset.step === stepId);
    }

    const base = `${runDir}/steps/${stepId}`;
    els.stepLabelPrompt.textContent = `${base}/prompt.txt`;
    els.stepLabelModel.textContent = `${base}/model_output.txt`;

    const prompt = await fetchTextMaybe(`${base}/prompt.txt`);
    const model = await fetchTextMaybe(`${base}/model_output.txt`);
    const action = await fetchJsonMaybe(`${base}/action.json`);
    const obs = await fetchJsonMaybe(`${base}/observation.json`);

    setPre(els.promptTxt, prompt);
    setPre(els.modelTxt, model);
    els.actionJson.textContent = action ? jsonPretty(action) : "(missing)";
    els.obsJson.textContent = obs ? jsonPretty(obs) : "(missing)";

    // artifacts
    els.files.innerHTML = "";
    const entries = await listDir(base);
    const main = new Set(["prompt.txt","model_output.txt","action.json","observation.json"]);
    const extsImg = new Set(["png","jpg","jpeg","gif","webp","bmp"]);

    const files = entries
      .filter(e => e && !e.is_dir && !main.has(String(e.name)))
      .map(e => String(e.rel_path));

    for (const p of files){
      const ext = (p.split(".").pop() || "").toLowerCase();
      const item = document.createElement("div");
      item.className = "fileItem";
      item.innerHTML = `<div class="path">${p}</div>`;

      if (extsImg.has(ext)) {
        const img = document.createElement("img");
        img.className = "preview";
        img.style.maxHeight = "180px";
        img.style.objectFit = "contain";
        img.style.marginTop = "8px";
        img.src = api.file(p);
        img.alt = p;
        img.addEventListener("click", () => showModalForFile(p));
        item.appendChild(img);
      }
      const btn = document.createElement("button");
      btn.textContent = "Open";
      btn.addEventListener("click", () => showModalForFile(p));
      item.appendChild(btn);

      els.files.appendChild(item);
    }
    if (!files.length) {
      const msg = document.createElement("div");
      msg.className = "hint";
      msg.textContent = "No artifacts found for this step.";
      els.files.appendChild(msg);
    }
  }

  async function selectRowById(id){
    const row = state.rowById.get(id);
    if (!row) return;
    state.selectedId = id;
    state.selectedStep = null;

    els.empty.style.display = "none";
    els.content.style.display = "block";

    els.rowIdx.textContent = `${row.__idx} (line ${row.__line})`;
    els.rowSplit.textContent = String(row.split || "");
    els.rowImage.textContent = String(row.image_path || "");
    els.rowRun.textContent = String(row.run_dir || "");
    els.rowTrue.textContent = normalizeLabelList(row.y_true).join(", ");
    els.rowPred.textContent = normalizeLabelList(row.y_pred).join(", ");
    els.rowErr.textContent = String(row.error || "");
    els.rowRaw.textContent = jsonPretty(row);

    const trace = { tool_sequence: row.tool_sequence || null, trace: row.trace || null };
    els.tracePre.textContent = jsonPretty(trace);

    // image: prefer run_dir/input.png
    const runDir = String(row.run_dir || "").trim();
    const imgPath = runDir ? `${runDir}/input.png` : String(row.image_path || "").trim();
    els.imgHint.textContent = "";
    els.img.onerror = () => {
      const fallback = String(row.image_path || "").trim();
      if (fallback && fallback !== imgPath) {
        els.img.src = api.file(fallback);
        els.imgHint.textContent = `source: ${fallback}`;
      } else {
        els.imgHint.textContent = "이미지 파일을 찾지 못했어요.";
      }
    };
    if (imgPath) {
      els.img.src = api.file(imgPath);
      els.imgHint.textContent = `source: ${imgPath}`;
    }

    await loadAndShowRunMeta(runDir);

    els.stepsBar.innerHTML = "";
    setPre(els.promptTxt, "(step 선택)");
    setPre(els.modelTxt, "(step 선택)");
    els.actionJson.textContent = "(step 선택)";
    els.obsJson.textContent = "(step 선택)";
    els.files.innerHTML = "";

    if (!runDir) {
      els.stepsBar.innerHTML = `<span class="hint">run_dir 없음</span>`;
      renderTable();
      return;
    }

    const entries = await listDir(`${runDir}/steps`);
    const stepIds = entries
      .filter(e => e && e.is_dir && /^step_\\d{3}$/.test(String(e.name)))
      .map(e => String(e.name))
      .sort();

    if (!stepIds.length) {
      els.stepsBar.innerHTML = `<span class="hint">step 폴더를 못 찾았어요.</span>`;
      renderTable();
      return;
    }

    for (const sid of stepIds){
      const btn = document.createElement("button");
      btn.className = "stepBtn";
      btn.textContent = sid;
      btn.dataset.step = sid;
      btn.addEventListener("click", () => renderStep(runDir, sid));
      els.stepsBar.appendChild(btn);
    }

    await renderStep(runDir, stepIds[0]);
    renderTable();
  }

  els.tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (!tr) return;
    const id = Number(tr.dataset.id);
    if (Number.isFinite(id)) selectRowById(id);
  });

  for (const el of [els.q, els.split, els.errOnly, els.hasRun, els.dedupe]){
    el.addEventListener("input", () => renderTable());
    el.addEventListener("change", () => renderTable());
  }

  async function init(){
    els.status.textContent = "loading rows…";
    const resp = await fetch(api.rows);
    const obj = await resp.json();
    state.rows = Array.isArray(obj.rows) ? obj.rows : [];
    state.rowById = new Map(state.rows.map(r => [r.__idx, r]));
    els.status.textContent = `predictions=${obj.n_rows} (${obj.predictions_path}) | base_dir=${obj.base_dir}`;
    renderTable();
  }

  init().catch(e => {
    els.status.textContent = "failed to load: " + String(e);
  });
})();
</script>
</body>
</html>
"""


def _write_json(handler: BaseHTTPRequestHandler, status: int, obj: Any) -> None:
    payload = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _write_text(handler: BaseHTTPRequestHandler, status: int, text: str, *, content_type: str) -> None:
    payload = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _write_bytes(handler: BaseHTTPRequestHandler, status: int, data: bytes, *, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _h(x: Any) -> str:
    return html.escape(str(x if x is not None else ""), quote=True)


def _qs_bool(qs: Dict[str, List[str]], key: str) -> bool:
    v = (qs.get(key) or [""])[0].strip().lower()
    return v in {"1", "true", "yes", "on"}


def _qs_str(qs: Dict[str, List[str]], key: str, default: str = "") -> str:
    return (qs.get(key) or [default])[0]


def _basename(path: str) -> str:
    p = str(path or "").replace("\\", "/")
    return p.split("/")[-1] if p else ""


def _run_id(run_dir: str) -> str:
    p = str(run_dir or "").replace("\\", "/").rstrip("/")
    return p.split("/")[-1] if p else ""


def _labels(x: Any) -> str:
    if isinstance(x, list):
        return ", ".join([str(v) for v in x])
    if isinstance(x, str):
        return x
    return ""


def _file_url(rel_path: str) -> str:
    return "/file?path=" + quote(str(rel_path or ""))


def _read_text_best_effort(app: "PredictionsWebApp", rel_path: str, *, max_chars: int = 200_000) -> Optional[str]:
    try:
        data, _ctype = app.read_file(rel_path)
    except Exception:
        return None
    try:
        s = data.decode("utf-8", errors="replace")
    except Exception:
        s = str(data[: max_chars])
    if max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars] + "\n\n...(truncated)..."
    return s


def _read_json_pretty(app: "PredictionsWebApp", rel_path: str, *, max_chars: int = 200_000) -> Optional[str]:
    t = _read_text_best_effort(app, rel_path, max_chars=max_chars)
    if t is None:
        return None
    try:
        obj = json.loads(t)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return t


_SSR_CSS = """
:root{
  --bg:#0b0f14; --panel:#0f1723; --muted:#9fb0c3; --fg:#e6edf3;
  --line:#223044; --accent:#4aa3ff; --bad:#ff5c5c; --good:#35d07f;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif;
}
*{box-sizing:border-box}
body{margin:0; font-family:var(--sans); background:var(--bg); color:var(--fg)}
a{color:var(--accent); text-decoration:none}
a:hover{text-decoration:underline}
header{padding:12px 14px; border-bottom:1px solid var(--line); background:linear-gradient(180deg,#0f1723, #0b0f14)}
header h1{margin:0 0 6px; font-size:16px}
.hint{margin-top:6px; font-size:12px; color:var(--muted); line-height:1.4}
.mono{font-family:var(--mono)}
.wrap{padding:14px}
.card{background:rgba(15,23,35,0.6); border:1px solid var(--line); border-radius:12px; padding:12px; margin-bottom:12px}
.grid{display:grid; grid-template-columns: 360px 1fr; gap:12px}
img.preview{max-width:100%; border-radius:10px; border:1px solid var(--line); background:#000}
table{width:100%; border-collapse:collapse; font-size:12px}
table a{color:inherit; text-decoration:none}
table a:hover{color:var(--accent); text-decoration:underline}
thead th{background:rgba(15,23,35,0.95); border-bottom:1px solid var(--line); text-align:left; padding:10px 10px; position:sticky; top:0}
tbody td{border-bottom:1px solid rgba(34,48,68,0.6); padding:8px 10px; vertical-align:top}
tbody tr:hover{background:rgba(74,163,255,0.08)}
.rowsTable tbody td a{display:block; width:100%; height:100%; padding:8px 10px; margin:-8px -10px; cursor:pointer}
.pill{display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid var(--line); color:var(--muted); font-size:11px}
.pill.bad{border-color:rgba(255,92,92,0.5); color:var(--bad)}
.pill.good{border-color:rgba(53,208,127,0.5); color:var(--good)}
pre{
  margin:0; padding:10px; border-radius:10px; background:rgba(11,15,20,0.8);
  border:1px solid rgba(34,48,68,0.8); overflow:auto; max-height:380px;
  font-family:var(--mono); font-size:11px; line-height:1.35; white-space:pre-wrap;
}
.formRow{display:flex; gap:10px; flex-wrap:wrap; align-items:end}
label{font-size:12px; color:var(--muted)}
input[type="text"], select{
  background:var(--panel); border:1px solid var(--line); color:var(--fg);
  padding:8px; border-radius:8px; outline:none;
}
.btn{
  display:inline-block; border:1px solid var(--line); background:rgba(74,163,255,0.12); color:var(--fg);
  padding:8px 12px; border-radius:10px; cursor:pointer;
}
.steps{display:flex; flex-wrap:wrap; gap:6px}
.step{display:inline-block; border:1px solid var(--line); background:rgba(15,23,35,0.6); color:var(--fg); padding:6px 10px; border-radius:10px; font-size:12px}
.step.active{border-color:rgba(74,163,255,0.9); box-shadow:0 0 0 2px rgba(74,163,255,0.15) inset}
.files{display:grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap:10px}
.fileItem{border:1px solid var(--line); background:rgba(15,23,35,0.45); border-radius:12px; padding:10px}
.fileItem .path{font-family:var(--mono); font-size:11px; color:var(--muted); word-break:break-all}
"""


def _page_html(*, title: str, body_html: str) -> str:
    return (
        "<!doctype html><html lang='ko'><head>"
        "<meta charset='utf-8' />"
        "<meta name='viewport' content='width=device-width,initial-scale=1' />"
        f"<title>{_h(title)}</title>"
        f"<style>{_SSR_CSS}</style>"
        "</head><body>"
        + body_html
        + "</body></html>"
    )


def _filter_rows(app: PredictionsWebApp, qs: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    rows = list(app.rows)

    if _qs_bool(qs, "dedupe"):
        m: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            key = str(r.get("image_path") or "").strip()
            if not key:
                key = f"__no_image__:{r.get('__idx')}"
            if key in m:
                m.pop(key, None)
            m[key] = r
        rows = list(m.values())

    split = _qs_str(qs, "split", "").strip().lower()
    q = _qs_str(qs, "q", "").strip().lower()
    err_only = _qs_bool(qs, "err")
    has_run = _qs_bool(qs, "run")

    def match(r: Dict[str, Any]) -> bool:
        if split and str(r.get("split") or "").strip().lower() != split:
            return False
        if has_run and not str(r.get("run_dir") or "").strip():
            return False
        has_err = bool(str(r.get("error") or "").strip())
        if err_only and not has_err:
            return False
        if q:
            hay = " ".join(
                [
                    str(r.get("image_path") or ""),
                    str(r.get("error") or ""),
                    str(r.get("run_dir") or ""),
                    str(r.get("split") or ""),
                    _labels(r.get("y_true")),
                    _labels(r.get("y_pred")),
                ]
            ).lower()
            if q not in hay:
                return False
        return True

    return [r for r in rows if match(r)]


def _render_index_page(app: PredictionsWebApp, qs: Dict[str, List[str]]) -> str:
    rows = _filter_rows(app, qs)
    q = _qs_str(qs, "q", "")
    split = _qs_str(qs, "split", "")
    err = _qs_bool(qs, "err")
    run = _qs_bool(qs, "run")
    dedupe = _qs_bool(qs, "dedupe")

    body = [
        "<header>",
        "<h1>Predictions / Run Viewer</h1>",
        "<div class='hint'>서버 렌더링 모드(브라우저 JS 없이도 동작). "
        "원하면 <a href='/spa'>SPA 모드</a>도 사용 가능.</div>",
        f"<div class='hint mono'>predictions: {_h(str(app.predictions_path))}<br/>base_dir: {_h(str(app.base_dir))}</div>",
        "</header>",
        "<div class='wrap'>",
        "<div class='card'>",
        "<form method='GET' action='/' class='formRow'>",
        "<div style='min-width:260px'>",
        "<label>검색 (image_path / error / run_id)</label><br/>",
        f"<input type='text' name='q' value='{_h(q)}' placeholder='dual axis / 20251222_203743 / Max steps' style='width:100%' />",
        "</div>",
        "<div style='min-width:140px'>",
        "<label>split</label><br/>",
        "<select name='split'>"
        f"<option value='' {'selected' if not split else ''}>all</option>"
        f"<option value='dev' {'selected' if split=='dev' else ''}>dev</option>"
        f"<option value='val' {'selected' if split=='val' else ''}>val</option>"
        "</select>",
        "</div>",
        "<div style='min-width:340px'>",
        "<label>옵션</label><br/>",
        "<div style='display:flex; gap:14px; align-items:center; padding:6px 0'>",
        f"<label><input type='checkbox' name='err' value='1' {'checked' if err else ''}/> error만</label>",
        f"<label><input type='checkbox' name='run' value='1' {'checked' if run else ''}/> run_dir만</label>",
        f"<label><input type='checkbox' name='dedupe' value='1' {'checked' if dedupe else ''}/> image_path dedupe(마지막)</label>",
        "</div>",
        "</div>",
        "<div>",
        f"<button class='btn' type='submit'>Apply</button>",
        "</div>",
        "</form>",
        f"<div class='hint'>showing {len(rows)} row(s) / total {len(app.rows)}</div>",
        "</div>",
        "<div class='card' style='padding:0'>",
        "<div style='overflow:auto; max-height:70vh'>",
        "<table class='rowsTable'>",
        "<thead><tr>"
        "<th style='width:64px'>idx</th>"
        "<th style='width:64px'>split</th>"
        "<th>image</th>"
        "<th style='width:140px'>y_true</th>"
        "<th style='width:140px'>y_pred</th>"
        "<th style='width:110px'>status</th>"
        "<th style='width:160px'>run_id</th>"
        "<th>error</th>"
        "</tr></thead>",
        "<tbody>",
    ]

    for r in rows:
        idx = int(r.get("__idx") or 0)
        href = f"/sample/{idx}"
        split_v = str(r.get("split") or "")
        image_path = str(r.get("image_path") or "")
        run_dir = str(r.get("run_dir") or "")
        run_id = _run_id(run_dir)
        err_s = str(r.get("error") or "").strip()
        status = "<span class='pill good'>ok</span>" if not err_s else "<span class='pill bad'>error</span>"
        body.append(
            "<tr>"
            f"<td class='mono'><a href='{href}'>{idx}</a></td>"
            f"<td><a href='{href}'>{_h(split_v)}</a></td>"
            f"<td title='{_h(image_path)}'><a href='{href}'>{_h(_basename(image_path))}</a></td>"
            f"<td class='mono'><a href='{href}'>{_h(_labels(r.get('y_true')))}</a></td>"
            f"<td class='mono'><a href='{href}'>{_h(_labels(r.get('y_pred')))}</a></td>"
            f"<td><a href='{href}'>{status}</a></td>"
            f"<td class='mono' title='{_h(run_dir)}'><a href='{href}'>{_h(run_id)}</a></td>"
            f"<td class='mono'><a href='{href}'>{_h(err_s[:180] + ('…' if len(err_s) > 180 else ''))}</a></td>"
            "</tr>"
        )

    body += [
        "</tbody></table></div></div>",
        "</div>",
    ]
    return _page_html(title="Predictions Viewer", body_html="".join(body))


def _render_sample_page(app: PredictionsWebApp, idx: int, qs: Dict[str, List[str]]) -> str:
    if idx < 0 or idx >= len(app.rows):
        return _page_html(title="Not Found", body_html="<header><h1>Not Found</h1></header><div class='wrap'>invalid idx</div>")

    row = app.rows[idx]
    run_dir = str(row.get("run_dir") or "").strip()
    image_path = str(row.get("image_path") or "").strip()

    # Best-effort run metadata (avoid crashing the page on malformed JSON).
    metadata_bundle: Optional[Dict[str, Any]] = None
    if run_dir:
        metadata_bundle = {}
        for key in ["metadata", "metadata_info", "metadata_error"]:
            rel = f"{run_dir}/{key}.json"
            t = _read_text_best_effort(app, rel)
            if t is None:
                metadata_bundle[key] = None
                continue
            try:
                metadata_bundle[key] = json.loads(t)
            except Exception as e:
                metadata_bundle[key] = {"__parse_error__": str(e), "__text__": t}

    # Choose image.
    img_rel = ""
    if run_dir and app.stat(f"{run_dir}/input.png"):
        img_rel = f"{run_dir}/input.png"
    elif image_path and app.stat(image_path):
        img_rel = image_path

    # Steps.
    step_names: List[str] = []
    if run_dir and app.stat(f"{run_dir}/steps"):
        try:
            entries = app.listdir(f"{run_dir}/steps")
            step_names = sorted([e.name for e in entries if e.is_dir and e.name.startswith("step_")])
        except Exception:
            step_names = []

    step = _qs_str(qs, "step", "").strip()
    if not step and step_names:
        step = step_names[0]
    if step and step not in set(step_names):
        step = step_names[0] if step_names else ""

    prev_idx = idx - 1 if idx > 0 else None
    next_idx = idx + 1 if idx + 1 < len(app.rows) else None

    body: List[str] = [
        "<header>",
        f"<h1>Sample idx={idx}</h1>",
        "<div class='hint'>"
        f"<a href='/'>← Back</a> "
        + (f"| <a href='/sample/{prev_idx}'>Prev</a>" if prev_idx is not None else "")
        + (f" | <a href='/sample/{next_idx}'>Next</a>" if next_idx is not None else ""),
        "</div>",
        "</header>",
        "<div class='wrap'>",
        "<div class='grid'>",
        "<div class='card'>",
        "<h2 style='margin:0 0 8px; font-size:13px; color:var(--muted)'>Image</h2>",
    ]
    if img_rel:
        body += [
            f"<img class='preview' src='{_file_url(img_rel)}' alt='image' />",
            f"<div class='hint mono'>source: {_h(img_rel)}</div>",
        ]
    else:
        body += ["<div class='hint'>이미지 파일을 찾지 못했어요.</div>"]
    body += ["</div>"]

    # Row info
    err_s = str(row.get("error") or "").strip()
    body += [
        "<div class='card'>",
        "<h2 style='margin:0 0 8px; font-size:13px; color:var(--muted)'>Row</h2>",
        "<table><tbody>",
        f"<tr><td class='mono'>split</td><td>{_h(row.get('split'))}</td></tr>",
        f"<tr><td class='mono'>image_path</td><td class='mono'>{_h(image_path)}</td></tr>",
        f"<tr><td class='mono'>run_dir</td><td class='mono'>{_h(run_dir)}</td></tr>",
        f"<tr><td class='mono'>y_true</td><td class='mono'>{_h(_labels(row.get('y_true')))}</td></tr>",
        f"<tr><td class='mono'>y_pred</td><td class='mono'>{_h(_labels(row.get('y_pred')))}</td></tr>",
        f"<tr><td class='mono'>error</td><td class='mono'>{_h(err_s)}</td></tr>",
        "</tbody></table>",
        "<details style='margin-top:10px'>"
        "<summary style='cursor:pointer; color:var(--muted); font-size:12px'>raw row JSON 보기</summary>"
        f"<pre>{_h(json.dumps(row, ensure_ascii=False, indent=2))}</pre>"
        "</details>",
        "</div>",
        "</div>",  # grid
    ]

    # Run metadata
    body += [
        "<div class='card'>",
        "<h2 style='margin:0 0 8px; font-size:13px; color:var(--muted)'>Run metadata</h2>",
        "<div class='grid' style='grid-template-columns: 1fr 1fr'>",
        "<div>",
        f"<div class='hint mono'>{_h(run_dir + '/answer.json' if run_dir else 'answer.json')}</div>",
        f"<pre>{_h(_read_json_pretty(app, f'{run_dir}/answer.json') or '(missing)') if run_dir else '(no run_dir)'}</pre>",
        "</div>",
        "<div>",
        "<div class='hint mono'>metadata.json / metadata_info.json / metadata_error.json</div>",
        f"<pre>{_h(json.dumps(metadata_bundle, ensure_ascii=False, indent=2) if metadata_bundle is not None else '(no run_dir)')}</pre>",
        "</div>",
        "</div>",
        "</div>",
    ]

    # Steps
    body += [
        "<div class='card'>",
        "<h2 style='margin:0 0 8px; font-size:13px; color:var(--muted)'>Steps</h2>",
    ]
    if not run_dir:
        body += ["<div class='hint'>run_dir 없음</div>"]
    elif not step_names:
        body += ["<div class='hint'>step 폴더를 못 찾았어요. (out/chartagent_runs 가 존재하는지 확인)</div>"]
    else:
        body.append("<div class='steps'>")
        for s in step_names:
            cls = "step active" if s == step else "step"
            body.append(f"<a class='{cls}' href='/sample/{idx}?step={_h(s)}'>{_h(s)}</a>")
        body.append("</div>")

        if step:
            base = f"{run_dir}/steps/{step}"
            prompt = _read_text_best_effort(app, f"{base}/prompt.txt")
            model = _read_text_best_effort(app, f"{base}/model_output.txt")
            action = _read_json_pretty(app, f"{base}/action.json")
            obs = _read_json_pretty(app, f"{base}/observation.json")

            body += [
                "<div class='grid' style='grid-template-columns:1fr 1fr; margin-top:12px'>",
                "<div>",
                f"<div class='hint mono'>{_h(base + '/prompt.txt')}</div>",
                f"<pre>{_h(prompt or '(missing)')}</pre>",
                "</div>",
                "<div>",
                f"<div class='hint mono'>{_h(base + '/model_output.txt')}</div>",
                f"<pre>{_h(model or '(missing)')}</pre>",
                "</div>",
                "</div>",
                "<div class='grid' style='grid-template-columns:1fr 1fr; margin-top:12px'>",
                "<div>",
                f"<div class='hint mono'>{_h(base + '/action.json')}</div>",
                f"<pre>{_h(action or '(missing)')}</pre>",
                "</div>",
                "<div>",
                f"<div class='hint mono'>{_h(base + '/observation.json')}</div>",
                f"<pre>{_h(obs or '(missing)')}</pre>",
                "</div>",
                "</div>",
            ]

            # Artifacts
            body.append("<div style='margin-top:12px'>")
            body.append("<div class='hint'>Artifacts</div>")
            body.append("<div class='files'>")
            try:
                entries = app.listdir(base)
            except Exception:
                entries = []
            skip = {"prompt.txt", "model_output.txt", "action.json", "observation.json"}
            for e in entries:
                if e.is_dir or e.name in skip:
                    continue
                rel = e.rel_path
                ext = os.path.splitext(e.name)[1].lower()
                is_img = ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
                body.append("<div class='fileItem'>")
                body.append(f"<div class='path'>{_h(rel)}</div>")
                if is_img:
                    body.append(f"<a href='{_file_url(rel)}' target='_blank'><img class='preview' src='{_file_url(rel)}' alt='{_h(rel)}' style='max-height:180px; object-fit:contain; margin-top:8px' /></a>")
                body.append(f"<div style='margin-top:8px'><a class='btn' href='{_file_url(rel)}' target='_blank'>Open</a></div>")
                body.append("</div>")
            body.append("</div></div>")

    body.append("</div>")  # card

    # Trace
    trace_obj = {"tool_sequence": row.get("tool_sequence"), "trace": row.get("trace")}
    body += [
        "<div class='card'>",
        "<h2 style='margin:0 0 8px; font-size:13px; color:var(--muted)'>Trace (JSONL)</h2>",
        f"<pre>{_h(json.dumps(trace_obj, ensure_ascii=False, indent=2))}</pre>",
        "</div>",
        "</div>",  # wrap
    ]

    return _page_html(title=f"Sample {idx}", body_html="".join(body))


def serve(*, predictions_path: str, base_dir: str, host: str, port: int, quiet: bool) -> None:
    app = PredictionsWebApp(predictions_path=predictions_path, base_dir=base_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            if quiet:
                return
            super().log_message(format, *args)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            try:
                if path == "/favicon.ico":
                    return _write_bytes(self, 204, b"", content_type="image/x-icon")

                if path in {"/", "/index.html"}:
                    html_text = _render_index_page(app, qs)
                    return _write_text(self, 200, html_text, content_type="text/html; charset=utf-8")

                if path == "/spa":
                    return _write_text(self, 200, INDEX_HTML, content_type="text/html; charset=utf-8")

                if path.startswith("/sample/"):
                    parts = [p for p in path.split("/") if p]
                    if len(parts) >= 2 and parts[0] == "sample":
                        try:
                            idx = int(parts[1])
                        except Exception:
                            return _write_text(self, 400, "invalid idx", content_type="text/plain; charset=utf-8")
                        html_text = _render_sample_page(app, idx, qs)
                        return _write_text(self, 200, html_text, content_type="text/html; charset=utf-8")

                if path == "/api/rows":
                    return _write_json(
                        self,
                        200,
                        {
                            "predictions_path": str(app.predictions_path),
                            "base_dir": str(app.base_dir),
                            "n_rows": len(app.rows),
                            "rows": app.rows,
                        },
                    )

                if path == "/api/listdir":
                    rel = unquote((qs.get("path") or [""])[0])
                    entries = app.listdir(rel)
                    return _write_json(
                        self,
                        200,
                        {
                            "path": _safe_relpath(rel),
                            "entries": [e.__dict__ for e in entries],
                        },
                    )

                if path == "/api/stat":
                    rel = unquote((qs.get("path") or [""])[0])
                    st = app.stat(rel)
                    return _write_json(self, 200, {"path": _safe_relpath(rel), "stat": st})

                if path == "/file":
                    rel = unquote((qs.get("path") or [""])[0])
                    data, ctype = app.read_file(rel)
                    return _write_bytes(self, 200, data, content_type=ctype)

                return _write_json(self, 404, {"error": "not_found", "path": path})
            except PermissionError as e:
                return _write_json(self, 403, {"error": "forbidden", "detail": str(e)})
            except FileNotFoundError:
                return _write_json(self, 404, {"error": "file_not_found"})
            except NotADirectoryError:
                return _write_json(self, 400, {"error": "not_a_directory"})
            except IsADirectoryError:
                return _write_json(self, 400, {"error": "is_a_directory"})
            except Exception as e:
                return _write_json(self, 500, {"error": "internal_error", "detail": str(e)})

    server = ThreadingHTTPServer((host, int(port)), Handler)
    print(f"Serving on http://{host}:{port}")
    print(f"predictions={predictions_path}")
    print(f"base_dir={base_dir}")
    print("Tip: if you're on a remote node, use ssh port forwarding (e.g., ssh -L 8000:localhost:8000 ...).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Local web viewer for ChartAgent predictions + run_dir steps.")
    ap.add_argument("--predictions", required=True, help="Predictions JSONL (e.g., out/misviz_devval_predictions*.jsonl)")
    ap.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Filesystem root to serve files from (default: repo root).",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--quiet", action="store_true", help="Reduce request logs.")
    args = ap.parse_args()

    serve(
        predictions_path=str(args.predictions),
        base_dir=str(args.base_dir),
        host=str(args.host),
        port=int(args.port),
        quiet=bool(args.quiet),
    )


if __name__ == "__main__":
    main()
