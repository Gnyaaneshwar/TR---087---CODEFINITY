"""
web/app.py
===========
Flask backend for the Robotic Grasp Planning Dashboard.

Routes
------
GET  /                  -- Serve the main UI
POST /api/run           -- Upload RGB+depth, run pipeline, return job ID
GET  /api/status/<id>   -- Poll job status + step logs
GET  /api/result/<id>   -- Return final JSON results
GET  /api/image/<id>    -- Serve the visualization PNG
GET  /api/samples       -- Use built-in sample images
"""

from __future__ import annotations
import os
import sys
import uuid
import json
import base64
import threading
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template

# ── Ensure project root is on sys.path ──────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from grasp_pipeline.pipeline import run_pipeline
from grasp_pipeline.utils.camera_intrinsics import CameraIntrinsics

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024   # 32 MB upload limit

UPLOAD_DIR = os.path.join(ROOT, "web", "uploads")
RESULT_DIR = os.path.join(ROOT, "web", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# In-memory job store
JOBS: dict = {}   # job_id -> {status, logs, result, error}

STEP_LABELS = [
    "Input Validation",
    "Point Cloud Generation",
    "Scene Segmentation",
    "Grasp Candidate Generation",
    "Initial Scoring",
    "Collision Checking",
    "Stability Estimation",
    "Hybrid Re-Ranking",
    "Physics Validation",
    "Output Generation",
    "Visualization",
]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples")
def run_samples():
    """Run the pipeline on the built-in sample images."""
    job_id = str(uuid.uuid4())[:8]
    rgb_path   = os.path.join(ROOT, "data", "sample_rgb.png")
    depth_path = os.path.join(ROOT, "data", "sample_depth.png")

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        # Generate them on first run
        _generate_samples()

    JOBS[job_id] = {"status": "queued", "logs": [], "result": None, "error": None}
    t = threading.Thread(target=_run_job, args=(job_id, rgb_path, depth_path), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/run", methods=["POST"])
def run_custom():
    """Accept uploaded RGB + depth images and run the pipeline."""
    if "rgb" not in request.files or "depth" not in request.files:
        return jsonify({"error": "Both 'rgb' and 'depth' files are required."}), 400

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    rgb_path   = os.path.join(job_dir, "rgb.png")
    depth_path = os.path.join(job_dir, "depth.png")
    request.files["rgb"].save(rgb_path)
    request.files["depth"].save(depth_path)

    JOBS[job_id] = {"status": "queued", "logs": [], "result": None, "error": None}
    t = threading.Thread(target=_run_job, args=(job_id, rgb_path, depth_path), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job ID"}), 404
    return jsonify({
        "status": job["status"],
        "logs":   job["logs"],
        "error":  job["error"],
    })


@app.route("/api/result/<job_id>")
def job_result(job_id):
    job = JOBS.get(job_id)
    if not job or job["result"] is None:
        return jsonify({"error": "Result not ready"}), 404
    return jsonify(job["result"])


@app.route("/api/image/<job_id>")
def job_image(job_id):
    job = JOBS.get(job_id)
    if not job or not job.get("vis_path"):
        return jsonify({"error": "Image not ready"}), 404
    return send_file(job["vis_path"], mimetype="image/png")


# ── Background job runner ────────────────────────────────────────────────────

def _log(job_id, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    JOBS[job_id]["logs"].append({"time": ts, "msg": msg})


def _run_job(job_id: str, rgb_path: str, depth_path: str):
    JOBS[job_id]["status"] = "running"
    result_dir = os.path.join(RESULT_DIR, job_id)
    os.makedirs(result_dir, exist_ok=True)

    # Monkey-patch print so we can capture step logs
    import builtins
    original_print = builtins.print

    def capturing_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        _log(job_id, msg)
        original_print(*args, **kwargs)

    builtins.print = capturing_print

    try:
        _log(job_id, "Pipeline starting...")
        result = run_pipeline(
            rgb_input    = rgb_path,
            depth_input  = depth_path,
            output_dir   = result_dir,
            n_candidates = 50,
            top_k        = 10,
        )

        # Read the visualization PNG as base64
        with open(result["vis_path"], "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        JOBS[job_id]["result"] = {
            **result["result_dict"],
            "vis_b64": img_b64,
            "timings": result["timings"],
        }
        JOBS[job_id]["vis_path"]  = result["vis_path"]
        JOBS[job_id]["status"]    = "done"
        _log(job_id, f"Done! Total time: {result['timings']['total']:.2f}s")

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"]  = traceback.format_exc()
        _log(job_id, f"ERROR: {e}")
    finally:
        builtins.print = original_print


def _generate_samples():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "generate_test_data",
        os.path.join(ROOT, "data", "generate_test_data.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rgb, depth_mm, _ = mod.generate_test_rgb_depth()
    mod.save_images(rgb, depth_mm)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Robotic Grasp Planning Dashboard")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=False, port=5000, threaded=True)
