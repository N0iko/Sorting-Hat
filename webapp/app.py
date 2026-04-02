# -*- coding: utf-8 -*-
"""
智能合约检测 Web App：输入合约地址，后台执行流水线，轮询获取检测结果。
"""

import os
import sys
import uuid
import threading

# 保证能导入项目根目录的 pipeline
FYP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FYP_ROOT not in sys.path:
    sys.path.insert(0, FYP_ROOT)

from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates")
app.config["JSON_AS_ASCII"] = False

# 后台任务状态：job_id -> { "status": "pending"|"done"|"error", "result": ... }
jobs = {}
jobs_lock = threading.Lock()


def run_job(job_id, address):
    # 确保在根目录执行，以便脚本能找到 vandal-master 等子目录
    os.chdir(FYP_ROOT)
    try:
        # 1. 动态导入你的流水线脚本 (假设文件名是 batch_detect.py)
        import batch_detect
        # 2. 强制重新加载，确保修改后的分数线等配置能立即生效
        import importlib
        importlib.reload(batch_detect)
        
        # 3. 调用单地址流水线函数
        # 该函数返回格式为: (success: bool, detection_dict | error_str)
        success, data = batch_detect.run_pipeline(address)
        
        with jobs_lock:
            if success:
                # data 此时是 detection 字典，包含 score, conclusion 等
                jobs[job_id] = {"status": "done", "result": data}
            else:
                # data 此时是错误字符串
                jobs[job_id] = {"status": "error", "result": {"error": data}}
                
    except Exception as e:
        import traceback
        print(f"Error detail: {traceback.format_exc()}") # 打印详细报错到后台
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
        return jsonify({"ok": False, "error": "请提供合约地址"}), 400
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
        return jsonify({"ok": False, "error": "任务不存在"}), 404
    return jsonify({
        "ok": True,
        "status": job["status"],
        "result": job.get("result"),
    })


if __name__ == "__main__":
    os.chdir(FYP_ROOT)  # 工作目录设为 FYP 根目录，便于 pipeline 找 vandal/GNN
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
