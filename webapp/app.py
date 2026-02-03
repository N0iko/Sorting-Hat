import os
import sys
import uuid
import threading


FYP_ROOT = os.getcwd()
if FYP_ROOT not in sys.path:
    sys.path.insert(0, FYP_ROOT)

from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates")
app.config["JSON_AS_ASCII"] = False


jobs = {}
jobs_lock = threading.Lock()


def run_job(job_id, address):
    os.chdir(FYP_ROOT)
    try:
        import pipeline
        success, data = pipeline.run_pipeline(address)
        with jobs_lock:
            if success:
                jobs[job_id] = {"status": "done", "result": data}
            else:
                jobs[job_id] = {"status": "error", "result": {"error": data}}
    except Exception as e:
        with jobs_lock:
            jobs[job_id] = {"status": "error", "result": {"error": str(e)}}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    address = (data.get("address") or request.form.get("address") or "").strip()
    if not address:
        return jsonify({"ok": False, "error": "Please provide the contract address"}), 400
    if not address.startswith("0x"):
        address = "0x" + address

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "pending", "result": None}

    thread = threading.Thread(target=run_job, args=(job_id, address))
    thread.daemon = True
    thread.start()

    return jsonify({"ok": True, "job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Task not found"}), 404
    return jsonify({
        "ok": True,
        "status": job["status"],
        "result": job.get("result"),
    })


if __name__ == "__main__":
    os.chdir(FYP_ROOT)  
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
