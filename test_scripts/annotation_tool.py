"""
Annotation Review Tool
======================
Run the auto-annotation pipeline on videos, review annotated frames,
and save approved ones (original frame + label) to class folders.

Usage:
    .\\backend\\venv\\Scripts\\python.exe test_scripts\\annotation_tool.py
    Open http://localhost:7860
"""

import os
import sys
import uuid
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPT  = ROOT / "test_scripts" / "auto_pipeline.py"
VISUALIZE_SCRIPT = ROOT / "test_scripts" / "visualize_labels.py"
JOBS_DIR         = ROOT / "annotation_jobs"
CLASSES_DIR      = ROOT / "annotatted_classes"
MODEL_PATH       = ROOT / "models" / "best_03062026.pt"
PYTHON           = ROOT / "backend" / "venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

# ── In-memory jobs ────────────────────────────────────────────────────────────
_jobs: dict = {}
# { job_id: { status, video, output_dir, logs[], saved[], skipped[] } }

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Annotation Tool")


# ── Background pipeline runner ────────────────────────────────────────────────
def _run_pipeline(job_id: str, videos_dir: Path, output_dir: Path,
                  classes: str, skip_training: bool,
                  conf: str = "0.35", fps: str = "2.0",
                  labeler: str = "grounding_dino", device: str = "0",
                  yolo_model: str = ""):
    job = _jobs[job_id]
    log = lambda m: job["logs"].append(m)

    try:
        cmd = [str(PYTHON), str(PIPELINE_SCRIPT),
               "--videos",  str(videos_dir),
               "--output",  str(output_dir),
               "--classes", classes,
               "--conf",    conf,
               "--fps",     fps,
               "--labeler", labeler,
               "--device",  device]
        if skip_training:
            cmd.append("--skip-training")
        if labeler == "yolo" and yolo_model:
            cmd += ["--yolo-model", yolo_model]

        log("▶ " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", cwd=str(ROOT)
        )
        for line in proc.stdout:
            log(line.rstrip())
        proc.wait()

        if proc.returncode != 0:
            job["status"] = "error"
            log(f"❌ Pipeline exited with code {proc.returncode}")
            return

        log("\n▶ Running visualize_labels...")
        proc2 = subprocess.Popen(
            [str(PYTHON), str(VISUALIZE_SCRIPT),
             "--pipeline-output", str(output_dir)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", cwd=str(ROOT)
        )
        for line in proc2.stdout:
            log(line.rstrip())
        proc2.wait()

        job["status"] = "ready"
        log("✓ Ready for review!")
    except Exception as exc:
        job["status"] = "error"
        log(f"Exception: {exc}")


def _run_visualize_only(job_id: str, output_dir: Path):
    job = _jobs[job_id]
    log = lambda m: job["logs"].append(m)
    try:
        log(f"▶ Running visualize_labels on {output_dir} ...")
        proc = subprocess.Popen(
            [str(PYTHON), str(VISUALIZE_SCRIPT),
             "--pipeline-output", str(output_dir)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", cwd=str(ROOT)
        )
        for line in proc.stdout:
            log(line.rstrip())
        proc.wait()
        job["status"] = "ready"
        log("✓ Ready for review!")
    except Exception as exc:
        job["status"] = "error"
        log(f"Exception: {exc}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/api/jobs")
async def create_job(
    video: Optional[UploadFile] = File(default=None),
    video_path: str = Form(default=""),
    existing_output: str = Form(default=""),
    classes: str = Form(default=""),
    skip_training: str = Form(default="true"),
    conf: str = Form(default="0.35"),
    fps: str = Form(default="1.0"),
    labeler: str = Form(default="yolo"),
    device: str = Form(default="0"),
    yolo_model: str = Form(default=""),
):
    job_id  = uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id

    # ── Mode: load existing pipeline output ──────────────────────────────────
    if existing_output.strip():
        out_dir = Path(existing_output.strip())
        if not (out_dir / "2_filtered_frames").exists():
            raise HTTPException(400, f"No 2_filtered_frames found under: {out_dir}")
        _jobs[job_id] = {
            "status": "running",
            "video": out_dir.name,
            "output_dir": str(out_dir),
            "logs": [],
            "saved": [],
            "skipped": [],
        }
        threading.Thread(
            target=_run_visualize_only, args=(job_id, out_dir), daemon=True
        ).start()
        return {"job_id": job_id}

    # ── Mode: run full pipeline ───────────────────────────────────────────────
    vids_dir = job_dir / "input"
    out_dir  = job_dir / "output"
    vids_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if video and video.filename:
        dest = vids_dir / video.filename
        with open(dest, "wb") as f:
            f.write(await video.read())
        video_name = video.filename
    elif video_path.strip():
        src = Path(video_path.strip())
        if not src.exists():
            raise HTTPException(400, f"File not found: {video_path}")
        dest = vids_dir / src.name
        try:
            os.link(src, dest)   # fast hard-link (same drive)
        except OSError:
            shutil.copy2(src, dest)
        video_name = src.name
    else:
        raise HTTPException(400, "Provide a video path, file upload, or existing output dir")

    _jobs[job_id] = {
        "status": "running",
        "video": video_name,
        "output_dir": str(out_dir),
        "logs": [],
        "saved": [],
        "skipped": [],
    }
    threading.Thread(
        target=_run_pipeline,
        args=(job_id, vids_dir, out_dir, classes, skip_training == "true",
              conf, fps, labeler, device, yolo_model),
        daemon=True
    ).start()
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs():
    return [
        {"job_id": k, "video": v["video"], "status": v["status"],
         "saved": len(v["saved"]), "skipped": len(v["skipped"])}
        for k, v in reversed(list(_jobs.items()))
    ]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, log_since: int = 0):
    if job_id not in _jobs:
        raise HTTPException(404)
    j  = _jobs[job_id]
    ad = Path(j["output_dir"]) / "5_annotated"
    total = sum(1 for f in ad.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")) if ad.exists() else 0
    return {
        "status":    j["status"],
        "video":     j["video"],
        "logs":      j["logs"][log_since:],
        "log_count": len(j["logs"]),
        "saved":     len(j["saved"]),
        "skipped":   len(j["skipped"]),
        "total":     total,
    }


@app.get("/api/jobs/{job_id}/images")
def get_images(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404)
    j  = _jobs[job_id]
    ad = Path(j["output_dir"]) / "5_annotated"
    if not ad.exists():
        return {"images": []}
    saved   = set(j["saved"])
    skipped = set(j["skipped"])
    images  = []
    for f in sorted(ad.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            st = ("saved"   if f.name in saved   else
                  "skipped" if f.name in skipped else "pending")
            images.append({"name": f.name, "status": st})
    return {"images": images}


@app.get("/api/jobs/{job_id}/image/{filename}")
def serve_image(job_id: str, filename: str):
    if job_id not in _jobs:
        raise HTTPException(404)
    path = Path(_jobs[job_id]["output_dir"]) / "5_annotated" / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/classes")
def list_classes():
    CLASSES_DIR.mkdir(exist_ok=True)
    return {"classes": sorted(d.name for d in CLASSES_DIR.iterdir() if d.is_dir())}


@app.get("/api/model-classes")
def model_classes():
    if not MODEL_PATH.exists():
        raise HTTPException(404, f"Model not found: {MODEL_PATH}")
    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        names = [model.names[i] for i in sorted(model.names)]
        return {"classes": names, "model": MODEL_PATH.name, "path": str(MODEL_PATH)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/classes")
def create_class(name: str = Form(...)):
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name required")
    (CLASSES_DIR / name).mkdir(parents=True, exist_ok=True)
    return {"ok": True, "name": name}


@app.post("/api/save")
def save_frame(
    job_id:     str = Form(...),
    filename:   str = Form(...),
    class_name: str = Form(...),
):
    if job_id not in _jobs:
        raise HTTPException(404)
    j    = _jobs[job_id]
    out  = Path(j["output_dir"])
    stem = Path(filename).stem
    images_dir   = CLASSES_DIR / class_name / "images"
    labels_dir   = CLASSES_DIR / class_name / "labels"
    annotated_dir= CLASSES_DIR / class_name / "annotated"
    for d in (images_dir, labels_dir, annotated_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Original frame → images/
    for src in [out / "2_filtered_frames" / filename,
                out / "2_filtered_frames" / (stem + ".png")]:
        if src.exists():
            shutil.copy2(src, images_dir / src.name)
            break
    # YOLO label → labels/
    label_src = out / "3_labels" / f"{stem}.txt"
    if label_src.exists():
        shutil.copy2(label_src, labels_dir / f"{stem}.txt")
    # Annotated (visualized) image → annotated/
    ann_src = out / "5_annotated" / filename
    if ann_src.exists():
        shutil.copy2(ann_src, annotated_dir / filename)

    if filename not in j["saved"]:
        j["saved"].append(filename)
    j["skipped"] = [s for s in j["skipped"] if s != filename]
    return {"ok": True}


@app.post("/api/skip")
def skip_frame(job_id: str = Form(...), filename: str = Form(...)):
    if job_id not in _jobs:
        raise HTTPException(404)
    j = _jobs[job_id]
    if filename not in j["skipped"]:
        j["skipped"].append(filename)
    j["saved"] = [s for s in j["saved"] if s != filename]
    return {"ok": True}


# ── Embedded HTML/CSS/JS ──────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotation Tool</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d0d14;color:#cbd5e1;min-height:100vh}

.topbar{background:#13131f;border-bottom:1px solid #1e2035;padding:12px 24px;display:flex;align-items:center;gap:10px}
.topbar-title{font-size:16px;font-weight:600;color:#f1f5f9}
.topbar-hint{font-size:11px;color:#334155;margin-left:auto;font-style:italic}

.main{padding:24px;max-width:1440px;margin:0 auto}

/* Home layout */
.home-grid{display:grid;grid-template-columns:380px 1fr;gap:20px;align-items:start}

/* Cards */
.card{background:#13131f;border:1px solid #1e2035;border-radius:10px;padding:20px}
.card-title{font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:16px}

/* Tabs */
.tabs{display:flex;border-bottom:1px solid #1e2035;margin-bottom:18px;gap:0}
.tab{padding:8px 16px;font-size:12px;cursor:pointer;color:#475569;border-bottom:2px solid transparent;margin-bottom:-1px;transition:color .15s}
.tab.active{color:#a5b4fc;border-bottom-color:#6366f1}
.tab-panel{display:none}.tab-panel.active{display:block}

/* Form */
.fg{margin-bottom:12px}
label{display:block;font-size:11px;color:#64748b;margin-bottom:4px}
input[type=text],input[type=file],select{width:100%;background:#080810;border:1px solid #1e2035;border-radius:6px;padding:7px 10px;color:#cbd5e1;font-size:13px;outline:none;transition:border-color .15s}
input[type=text]:focus,select:focus{border-color:#6366f1}
input[type=file]{padding:5px 10px;cursor:pointer}
.row{display:flex;align-items:center;gap:8px}
.row input[type=text]{flex:1}
input[type=checkbox]{accent-color:#6366f1;width:14px;height:14px;cursor:pointer}
.check-row{display:flex;align-items:center;gap:8px;font-size:12px;color:#94a3b8}

.divider{text-align:center;font-size:11px;color:#334155;margin:10px 0;position:relative}
.divider::before,.divider::after{content:"";position:absolute;top:50%;width:calc(50% - 16px);height:1px;background:#1e2035}
.divider::before{left:0}.divider::after{right:0}

/* Buttons */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500;padding:8px 16px;transition:opacity .15s,transform .1s;white-space:nowrap}
.btn:hover{opacity:.85}.btn:active{transform:scale(.98)}.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-primary{background:#6366f1;color:#fff}
.btn-success{background:#10b981;color:#fff}
.btn-danger{background:#ef4444;color:#fff}
.btn-warn{background:#f59e0b;color:#000}
.btn-ghost{background:#1e2035;color:#cbd5e1}
.btn-sm{padding:5px 11px;font-size:12px}
.btn-full{width:100%}

/* Badges */
.badge{display:inline-block;padding:2px 7px;border-radius:12px;font-size:11px;font-weight:500;white-space:nowrap}
.badge-running{background:#1d3252;color:#60a5fa}
.badge-ready{background:#14352a;color:#34d399}
.badge-error{background:#2d1414;color:#f87171}

/* Job list items */
.job-item{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:7px;cursor:pointer;transition:background .1s;border:1px solid transparent;margin-bottom:6px}
.job-item:hover{background:#1a1a2e;border-color:#2d2d4a}
.job-item.active-job{background:#1a1a2e;border-color:#6366f1}
.job-name{font-size:13px;color:#e2e8f0;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.job-meta{font-size:11px;color:#475569;white-space:nowrap}

/* Log terminal */
.log-box{background:#06060d;border:1px solid #1e2035;border-radius:6px;padding:10px;font-family:'Cascadia Code','Fira Code',monospace;font-size:11px;height:180px;overflow-y:auto;color:#64748b;line-height:1.6}
.log-ok{color:#34d399}.log-err{color:#f87171}

/* Progress */
.pbar{height:3px;background:#1e2035;border-radius:2px;overflow:hidden;margin:8px 0}
.pfill{height:100%;background:#6366f1;border-radius:2px;transition:width .4s}

/* Review */
.review-header{display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.review-stats{display:flex;gap:16px;font-size:12px;flex-wrap:wrap}
.stat{display:flex;align-items:center;gap:4px}

.filter-row{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap}
.fbtn{padding:4px 14px;border-radius:20px;font-size:12px;cursor:pointer;border:1px solid #1e2035;background:#13131f;color:#64748b;transition:all .15s}
.fbtn.on{border-color:#6366f1;color:#a5b4fc;background:#1a1a2e}

/* Image grid */
.img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
.img-card{background:#13131f;border:1px solid #1e2035;border-radius:8px;overflow:hidden;cursor:pointer;transition:border-color .15s,transform .15s;position:relative}
.img-card:hover{border-color:#4f46e5;transform:translateY(-2px)}
.img-card.c-saved{border-color:#10b981}
.img-card.c-skipped{border-color:#f59e0b;opacity:.55}
.img-card img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;background:#080810}
.img-foot{padding:6px 8px;display:flex;align-items:center;justify-content:space-between}
.img-fname{font-size:10px;color:#475569;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:130px}
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.dot-pending{background:#334155}.dot-saved{background:#10b981}.dot-skipped{background:#f59e0b}

/* Modal */
.mbd{position:fixed;inset:0;background:rgba(0,0,0,.88);display:flex;align-items:center;justify-content:center;z-index:100;padding:16px}
.mbd.hide{display:none}
.modal{background:#13131f;border:1px solid #1e2035;border-radius:12px;width:100%;max-width:960px;max-height:92vh;display:flex;flex-direction:column;overflow:hidden}
.mimg-wrap{flex:1;overflow:hidden;display:flex;align-items:center;justify-content:center;background:#06060d;min-height:200px}
.mimg-wrap img{max-width:100%;max-height:65vh;object-fit:contain}
.mfoot{padding:14px 18px;border-top:1px solid #1e2035;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.mfname{font-size:11px;color:#475569;flex:1;min-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mnav{display:flex;align-items:center;gap:6px;margin-left:auto}
.mnav-counter{font-size:11px;color:#475569;white-space:nowrap;min-width:60px;text-align:center}
.kbd{display:inline-block;background:#1e2035;border-radius:3px;padding:1px 5px;font-size:10px;color:#64748b;font-family:monospace}

/* New class input */
.new-cls-area{padding:0 18px 14px;display:none}
.new-cls-row{display:flex;gap:6px}
.new-cls-row input{flex:1}

::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:#0d0d14}
::-webkit-scrollbar-thumb{background:#1e2035;border-radius:4px}
</style>
</head>
<body>

<div class="topbar">
  <span class="topbar-title">🎯 Annotation Tool</span>
  <span class="topbar-hint">Pipeline → Review → Save</span>
</div>

<!-- ══ HOME VIEW ════════════════════════════════════════════════════════════ -->
<div class="main" id="home-view">
  <div class="home-grid">

    <!-- Left: new job form -->
    <div class="card">
      <div class="card-title">New Job</div>

      <!-- Tabs: New Pipeline / Existing Output -->
      <div class="tabs">
        <div class="tab active" onclick="switchTab('new',this)">New Pipeline</div>
        <div class="tab" onclick="switchTab('existing',this)">Load Existing Output</div>
      </div>

      <!-- Tab: new pipeline -->
      <div class="tab-panel active" id="tab-new">
        <div class="fg">
          <label>Local video path</label>
          <input type="text" id="video-path" placeholder="D:\yolo_world_poc\Test_Footage\test2.mp4">
        </div>
        <div class="divider">or upload file</div>
        <div class="fg">
          <label>Video file</label>
          <input type="file" id="video-file" accept="video/*">
        </div>
        <div class="fg">
          <label>Classes <span id="classes-hint" style="color:#6366f1;font-style:italic;margin-left:4px"></span></label>
          <input type="text" id="classes" placeholder="e.g. french_fries,curly_fries,burger">
        </div>
        <div class="fg">
          <div class="check-row">
            <input type="checkbox" id="skip-train" checked>
            <label for="skip-train" style="margin:0;cursor:pointer">Skip training step</label>
          </div>
        </div>

        <!-- Advanced config -->
        <details style="margin-bottom:14px">
          <summary style="font-size:11px;color:#64748b;cursor:pointer;user-select:none;list-style:none;display:flex;align-items:center;gap:6px">
            <span id="adv-arrow" style="font-size:10px">▶</span>
            <span style="text-transform:uppercase;letter-spacing:.08em">Advanced Config</span>
          </summary>
          <div style="margin-top:12px;display:flex;flex-direction:column;gap:10px" onclick="event.stopPropagation()">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
              <div>
                <label>Confidence threshold</label>
                <div style="display:flex;align-items:center;gap:8px">
                  <input type="range" id="conf" min="0.1" max="0.9" step="0.05" value="0.35"
                    style="flex:1;accent-color:#6366f1" oninput="document.getElementById('conf-val').textContent=this.value">
                  <span id="conf-val" style="font-size:12px;color:#a5b4fc;min-width:30px">0.35</span>
                </div>
              </div>
              <div>
                <label>FPS (frames/sec extracted)</label>
                <div style="display:flex;align-items:center;gap:8px">
                  <input type="range" id="fps" min="0.5" max="5" step="0.5" value="1"
                    style="flex:1;accent-color:#6366f1" oninput="document.getElementById('fps-val').textContent=this.value">
                  <span id="fps-val" style="font-size:12px;color:#a5b4fc;min-width:30px">1</span>
                </div>
              </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
              <div>
                <label>Labeler</label>
                <select id="labeler" onchange="onLabelerChange()">
                  <option value="yolo" selected>YOLO (custom model)</option>
                  <option value="grounding_dino">Grounding DINO</option>
                </select>
              </div>
              <div>
                <label>Device</label>
                <select id="device">
                  <option value="0">GPU (0)</option>
                  <option value="cpu">CPU</option>
                </select>
              </div>
            </div>
            <div id="yolo-model-row">
              <label>YOLO model path</label>
              <input type="text" id="yolo-model" value="">
            </div>
          </div>
        </details>
      </div>

      <!-- Tab: existing output -->
      <div class="tab-panel" id="tab-existing">
        <div class="fg">
          <label>Existing pipeline output directory</label>
          <input type="text" id="existing-path" placeholder="D:\yolo_world_poc\pipeline_output_yolo">
        </div>
        <p style="font-size:11px;color:#475569;margin-bottom:12px">
          Must contain a <code style="color:#94a3b8">2_filtered_frames/</code> folder.
          Will run visualize_labels and open for review.
        </p>
      </div>

      <button class="btn btn-primary btn-full" id="run-btn" onclick="submitJob()">
        ▶ Run
      </button>

      <!-- Active job progress -->
      <div id="active-panel" style="margin-top:16px;display:none">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
          <span style="font-size:12px;color:#94a3b8" id="ap-label"></span>
          <span class="badge badge-running" id="ap-badge">running</span>
        </div>
        <div class="pbar"><div class="pfill" id="pfill" style="width:20%"></div></div>
        <div class="log-box" id="log-box"></div>
        <div style="margin-top:8px">
          <button class="btn btn-success btn-sm" id="go-review-btn" style="display:none"
                  onclick="openReview(activeJobId)">Open Review →</button>
        </div>
      </div>
    </div>

    <!-- Right: jobs list -->
    <div class="card">
      <div class="card-title">Jobs</div>
      <div id="jobs-list"><div style="color:#334155;font-size:13px">No jobs yet.</div></div>
    </div>

  </div>
</div>

<!-- ══ REVIEW VIEW ══════════════════════════════════════════════════════════ -->
<div class="main" id="review-view" style="display:none">
  <div class="review-header">
    <button class="btn btn-ghost btn-sm" onclick="goHome()">← Back</button>
    <div>
      <div style="font-size:15px;font-weight:600;color:#f1f5f9" id="rv-title"></div>
      <div style="font-size:11px;color:#475569" id="rv-jobid"></div>
    </div>
    <div class="review-stats" id="rv-stats"></div>
    <button class="btn btn-ghost btn-sm" style="margin-left:auto" onclick="loadImages()">↻ Refresh</button>
  </div>

  <div class="filter-row">
    <button class="fbtn on" onclick="setFilter('all',this)">All</button>
    <button class="fbtn" onclick="setFilter('pending',this)">Pending</button>
    <button class="fbtn" onclick="setFilter('saved',this)">Saved</button>
    <button class="fbtn" onclick="setFilter('skipped',this)">Skipped</button>
  </div>

  <div class="img-grid" id="img-grid"></div>
</div>

<!-- ══ MODAL ════════════════════════════════════════════════════════════════ -->
<div class="mbd hide" id="mbd" onclick="closeMbd(event)">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="mimg-wrap">
      <img id="mimg" src="" alt="">
    </div>
    <div class="mfoot">
      <div class="mfname" id="mfname"></div>
      <select id="mclass" style="min-width:160px">
        <option value="">Select class…</option>
      </select>
      <button class="btn btn-ghost btn-sm" onclick="showNewCls()">+ New</button>
      <button class="btn btn-danger btn-sm" onclick="doSkip()" title="Skip (X)">Skip</button>
      <button class="btn btn-success btn-sm" onclick="doSave()" title="Save (S / Enter)">Save</button>
      <div class="mnav">
        <button class="btn btn-ghost btn-sm" onclick="nav(-1)">‹</button>
        <span class="mnav-counter" id="mcounter"></span>
        <button class="btn btn-ghost btn-sm" onclick="nav(1)">›</button>
      </div>
    </div>
    <div id="new-cls-area" class="new-cls-area">
      <div class="new-cls-row">
        <input type="text" id="new-cls-input" placeholder="New class name (e.g. curly_fries)">
        <button class="btn btn-primary btn-sm" onclick="createClass()">Create</button>
        <button class="btn btn-ghost btn-sm" onclick="hideNewCls()">✕</button>
      </div>
    </div>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let activeJobId = null;
let logOffset   = 0;
let pollTimer   = null;
let currentMode = 'new';   // 'new' | 'existing'

let reviewJobId = null;
let allImgs     = [];
let filtImgs    = [];
let curFilter   = 'all';
let curIdx      = 0;

// ── Tabs ───────────────────────────────────────────────────────────────────
function switchTab(mode, el) {
  currentMode = mode;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-' + mode).classList.add('active');
}

// ── Submit job ─────────────────────────────────────────────────────────────
async function submitJob() {
  const fd = new FormData();

  if (currentMode === 'existing') {
    const ep = document.getElementById('existing-path').value.trim();
    if (!ep) { alert('Enter the pipeline output directory path'); return; }
    fd.append('existing_output', ep);
  } else {
    const vpath = document.getElementById('video-path').value.trim();
    const vfile = document.getElementById('video-file').files[0];
    if (!vpath && !vfile) { alert('Provide a video path or upload a file'); return; }
    if (vfile) fd.append('video', vfile);
    else        fd.append('video_path', vpath);
    const isYolo = document.getElementById('labeler').value === 'yolo';
    fd.append('classes', isYolo ? 'placeholder' : document.getElementById('classes').value.trim());
    fd.append('skip_training', document.getElementById('skip-train').checked ? 'true' : 'false');
    fd.append('conf',          document.getElementById('conf').value);
    fd.append('fps',           document.getElementById('fps').value);
    fd.append('labeler',       document.getElementById('labeler').value);
    fd.append('device',        document.getElementById('device').value);
    fd.append('yolo_model',    document.getElementById('yolo-model').value.trim());
  }

  document.getElementById('run-btn').disabled = true;
  const res = await fetch('/api/jobs', { method: 'POST', body: fd });
  if (!res.ok) {
    const t = await res.text(); alert('Error: ' + t);
    document.getElementById('run-btn').disabled = false;
    return;
  }
  const { job_id } = await res.json();
  activeJobId = job_id;
  logOffset   = 0;

  const panel = document.getElementById('active-panel');
  panel.style.display = 'block';
  document.getElementById('ap-label').textContent = '';
  document.getElementById('ap-badge').textContent = 'running';
  document.getElementById('ap-badge').className   = 'badge badge-running';
  document.getElementById('pfill').style.width    = '20%';
  document.getElementById('log-box').innerHTML    = '';
  document.getElementById('go-review-btn').style.display = 'none';

  startPoll();
  refreshJobs();
}

// ── Poll active job ────────────────────────────────────────────────────────
function startPoll() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollJob, 2000);
  pollJob();
}

async function pollJob() {
  if (!activeJobId) return;
  const res = await fetch(`/api/jobs/${activeJobId}?log_since=${logOffset}`);
  if (!res.ok) return;
  const d = await res.json();

  const box = document.getElementById('log-box');
  (d.logs || []).forEach(line => {
    const el = document.createElement('div');
    el.className = line.startsWith('✓') || line.startsWith('▶') ? 'log-ok'
                 : line.includes('❌') || /error/i.test(line)   ? 'log-err' : '';
    el.textContent = line;
    box.appendChild(el);
  });
  logOffset = d.log_count;
  box.scrollTop = box.scrollHeight;

  document.getElementById('ap-label').textContent = d.video || '';

  if (d.status === 'ready') {
    document.getElementById('ap-badge').textContent = 'ready';
    document.getElementById('ap-badge').className   = 'badge badge-ready';
    document.getElementById('pfill').style.width    = '100%';
    document.getElementById('go-review-btn').style.display = 'inline-flex';
    document.getElementById('run-btn').disabled = false;
    clearInterval(pollTimer);
    refreshJobs();
  } else if (d.status === 'error') {
    document.getElementById('ap-badge').textContent = 'error';
    document.getElementById('ap-badge').className   = 'badge badge-error';
    document.getElementById('run-btn').disabled = false;
    clearInterval(pollTimer);
    refreshJobs();
  } else {
    const fill = document.getElementById('pfill');
    const cur  = parseFloat(fill.style.width) || 20;
    fill.style.width = Math.min(cur + 2, 90) + '%';
  }
}

// ── Jobs list ──────────────────────────────────────────────────────────────
async function refreshJobs() {
  const res = await fetch('/api/jobs');
  const jobs = await res.json();
  const list = document.getElementById('jobs-list');
  if (!jobs.length) {
    list.innerHTML = '<div style="color:#334155;font-size:13px">No jobs yet.</div>';
    return;
  }
  list.innerHTML = jobs.map(j => `
    <div class="job-item ${j.job_id === reviewJobId ? 'active-job' : ''}"
         onclick="openReview('${j.job_id}')">
      <div style="min-width:0;flex:1">
        <div class="job-name">${j.video}</div>
        <div style="font-size:10px;color:#334155;margin-top:2px">${j.job_id}</div>
      </div>
      <span class="badge badge-${j.status}">${j.status}</span>
      <div class="job-meta">
        <span style="color:#34d399">${j.saved}↗</span>
        <span style="color:#f59e0b;margin-left:6px">${j.skipped}✗</span>
      </div>
    </div>
  `).join('');
}

// ── Review view ────────────────────────────────────────────────────────────
function goHome() {
  reviewJobId = null;
  document.getElementById('home-view').style.display = 'block';
  document.getElementById('review-view').style.display = 'none';
  refreshJobs();
}

async function openReview(jobId) {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) { alert('Job not found'); return; }
  const job = await res.json();
  if (job.status === 'running') {
    alert('Pipeline is still running — wait for it to finish.'); return;
  }
  reviewJobId = jobId;
  document.getElementById('rv-title').textContent  = job.video;
  document.getElementById('rv-jobid').textContent  = jobId;
  document.getElementById('home-view').style.display   = 'none';
  document.getElementById('review-view').style.display = 'block';
  curFilter = 'all';
  document.querySelectorAll('.fbtn').forEach((b,i) => b.classList.toggle('on', i===0));
  await loadImages();
  await loadClasses();
}

async function loadImages() {
  if (!reviewJobId) return;
  const [ir, jr] = await Promise.all([
    fetch(`/api/jobs/${reviewJobId}/images`),
    fetch(`/api/jobs/${reviewJobId}`),
  ]);
  allImgs = (await ir.json()).images;
  const j = await jr.json();

  const saved   = allImgs.filter(i => i.status==='saved').length;
  const skipped = allImgs.filter(i => i.status==='skipped').length;
  const pending = allImgs.length - saved - skipped;

  document.getElementById('rv-stats').innerHTML = `
    <div class="stat"><span style="color:#34d399;font-weight:600">${saved}</span>&nbsp;saved</div>
    <div class="stat"><span style="color:#f59e0b;font-weight:600">${skipped}</span>&nbsp;skipped</div>
    <div class="stat"><span style="color:#94a3b8;font-weight:600">${pending}</span>&nbsp;pending</div>
    <div class="stat"><span style="color:#475569">${allImgs.length}</span>&nbsp;total</div>
  `;
  applyFilter(curFilter);
}

function setFilter(f, btn) {
  curFilter = f;
  document.querySelectorAll('.fbtn').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  applyFilter(f);
}

function applyFilter(f) {
  filtImgs = f === 'all' ? allImgs : allImgs.filter(i => i.status === f);
  renderGrid();
}

function renderGrid() {
  document.getElementById('img-grid').innerHTML = filtImgs.map((img, idx) => `
    <div class="img-card ${img.status==='saved'?'c-saved':img.status==='skipped'?'c-skipped':''}"
         onclick="openModal(${idx})">
      <img src="/api/jobs/${reviewJobId}/image/${encodeURIComponent(img.name)}" loading="lazy" alt="">
      <div class="img-foot">
        <span class="img-fname" title="${img.name}">${img.name}</span>
        <span class="dot dot-${img.status}"></span>
      </div>
    </div>
  `).join('');
}

// ── Modal ──────────────────────────────────────────────────────────────────
function openModal(idx) {
  curIdx = idx;
  renderModal();
  document.getElementById('mbd').classList.remove('hide');
}

function renderModal() {
  if (curIdx < 0 || curIdx >= filtImgs.length) return;
  const img = filtImgs[curIdx];
  document.getElementById('mimg').src     = `/api/jobs/${reviewJobId}/image/${encodeURIComponent(img.name)}`;
  document.getElementById('mfname').textContent  = img.name;
  document.getElementById('mcounter').textContent = `${curIdx+1} / ${filtImgs.length}`;
}

function closeMbd(e) {
  document.getElementById('mbd').classList.add('hide');
}

function nav(d) {
  const n = curIdx + d;
  if (n >= 0 && n < filtImgs.length) { curIdx = n; renderModal(); }
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (document.getElementById('mbd').classList.contains('hide')) return;
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft'  || e.key === 'a') nav(-1);
  if (e.key === 'ArrowRight' || e.key === 'd') nav(1);
  if (e.key === 's' || e.key === 'Enter')       doSave();
  if (e.key === 'x')                            doSkip();
  if (e.key === 'Escape')                       closeMbd();
});

async function doSave() {
  const img = filtImgs[curIdx];
  const cls = document.getElementById('mclass').value;
  if (!cls) { alert('Select a class first'); return; }
  const fd = new FormData();
  fd.append('job_id', reviewJobId);
  fd.append('filename', img.name);
  fd.append('class_name', cls);
  await fetch('/api/save', { method:'POST', body:fd });
  img.status = 'saved';
  // Advance to next pending image
  const next = filtImgs.findIndex((im,i) => i > curIdx && im.status==='pending');
  if (next !== -1) curIdx = next;
  else if (curIdx < filtImgs.length-1) curIdx++;
  renderModal();
  await loadImages(); applyFilter(curFilter); renderModal();
}

async function doSkip() {
  const img = filtImgs[curIdx];
  const fd = new FormData();
  fd.append('job_id', reviewJobId);
  fd.append('filename', img.name);
  await fetch('/api/skip', { method:'POST', body:fd });
  img.status = 'skipped';
  if (curIdx < filtImgs.length-1) curIdx++;
  renderModal();
  await loadImages(); applyFilter(curFilter); renderModal();
}

// ── Classes ────────────────────────────────────────────────────────────────
async function loadClasses() {
  const { classes } = await (await fetch('/api/classes')).json();
  const sel = document.getElementById('mclass');
  const cur = sel.value;
  sel.innerHTML = '<option value="">Select class…</option>' +
    classes.map(c => `<option value="${c}"${c===cur?' selected':''}>${c}</option>`).join('');
}

function showNewCls() {
  document.getElementById('new-cls-area').style.display = 'block';
  document.getElementById('new-cls-input').focus();
}
function hideNewCls() { document.getElementById('new-cls-area').style.display = 'none'; }

async function createClass() {
  const name = document.getElementById('new-cls-input').value.trim();
  if (!name) return;
  const fd = new FormData(); fd.append('name', name);
  await fetch('/api/classes', { method:'POST', body:fd });
  document.getElementById('new-cls-input').value = '';
  hideNewCls();
  await loadClasses();
  document.getElementById('mclass').value = name;
}

// ── Labeler toggle ─────────────────────────────────────────────────────────
function onLabelerChange() {
  const isYolo = document.getElementById('labeler').value === 'yolo';
  document.getElementById('yolo-model-row').style.display = isYolo ? 'block' : 'none';
  const classesEl = document.getElementById('classes');
  if (isYolo) {
    classesEl.disabled = true;
    classesEl.style.opacity = '0.4';
    document.getElementById('classes-hint').textContent = 'auto from YOLO model';
  } else {
    classesEl.disabled = false;
    classesEl.style.opacity = '1';
  }
}

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  refreshJobs();

  // Pre-fill yolo-model path with the known model
  try {
    const res = await fetch('/api/model-classes');
    if (res.ok) {
      const { classes, model, path: modelPath } = await res.json();
      document.getElementById('classes').value = classes.join(',');
      document.getElementById('classes-hint').textContent = `from ${model}`;
      document.getElementById('yolo-model').value = modelPath || '';
      return;
    }
  } catch (_) {}
  // Fallback: use existing class folder names
  const { classes } = await (await fetch('/api/classes')).json();
  if (classes.length) {
    document.getElementById('classes').value = classes.join(',');
    document.getElementById('classes-hint').textContent = 'from annotatted_classes folders';
  }
  onLabelerChange();  // apply correct initial state
}
init();
setInterval(refreshJobs, 8000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    JOBS_DIR.mkdir(exist_ok=True)
    print(f"Python  : {PYTHON}")
    print(f"Classes : {CLASSES_DIR}")
    print(f"Jobs    : {JOBS_DIR}")
    print("Open    : http://localhost:7860\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
