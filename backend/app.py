import json
import os
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from ppg_features import extract_ppg_features

import base64
import io
import joblib

app = Flask(__name__)
CORS(app)

QUALITY_MODEL = None
QUALITY_SCALER = None

MIN_SAMPLES_FOR_DETECTION = 150  
SAMPLES_TO_KEEP = 150  


def load_quality_model():
    global QUALITY_MODEL, QUALITY_SCALER
    if QUALITY_MODEL is not None:
        return
    try:
        import joblib
        model_path = os.path.join(os.path.dirname(__file__), "quality_model.joblib")
        scaler_path = os.path.join(os.path.dirname(__file__), "quality_scaler.joblib")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            QUALITY_MODEL = joblib.load(model_path)
            QUALITY_SCALER = joblib.load(scaler_path)
    except Exception:
        pass


DATA_FILE = os.path.join(os.path.dirname(__file__), "records.json")
LABELED_FILE = os.path.join(os.path.dirname(__file__), "labeled_records.json")


def load_records():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []


def save_records(records):
    with open(DATA_FILE, "w") as f:
        json.dump(records, f, indent=2)


def load_labeled():
    if os.path.exists(LABELED_FILE):
        with open(LABELED_FILE, "r") as f:
            return json.load(f)
    return []


def save_labeled(records):
    with open(LABELED_FILE, "w") as f:
        json.dump(records, f, indent=2)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True})


@app.route('/save-record', methods=['POST'])
def save_record():
    try:
        body = request.get_json()
        if not body:
            return jsonify({"success": False, "error": "No body"}), 400
        records = load_records()
        record = {
            "heartRate": body.get("heartRate", {}),
            "hrv": body.get("hrv", {}),
            "ppgData": body.get("ppgData", []),
            "timestamp": body.get("timestamp") or datetime.utcnow().isoformat(),
        }
        records.append(record)
        save_records(records)
        return jsonify({"success": True, "data": record}), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/save-labeled-segment', methods=['POST'])
def save_labeled_segment():
    try:
        body = request.get_json()
        if not body:
            return jsonify({"success": False, "error": "No body"}), 400
        ppg_data = body.get("ppgData", [])
        label = body.get("label")
        if not isinstance(ppg_data, list) or label not in ("good", "bad"):
            return jsonify(
                {"success": False, "error": "Need ppgData (array) and label (good/bad)"}
            ), 400
        records = load_labeled()
        records.append({
            "ppgData": ppg_data,
            "label": label,
            "timestamp": datetime.utcnow().isoformat(),
        })
        save_labeled(records)
        return jsonify({"success": True}), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/infer-quality', methods=['POST'])
def infer_quality():
    try:
        body = request.get_json()
        if not body or not body.get("ppgData"):
            return jsonify({"error": "Missing ppgData"}), 400
        ppg_data = body["ppgData"]
        if not isinstance(ppg_data, list) or len(ppg_data) < 10:
            return jsonify(
                {"error": "ppgData must be an array with at least 10 points"}
            ), 400
        load_quality_model()
        if QUALITY_MODEL is None or QUALITY_SCALER is None:
            return jsonify({
                "label": None,
                "confidence": 0,
                "message": "No model trained yet. Collect labeled segments and run train_quality_model.py.",
            }), 200
        features = extract_ppg_features(ppg_data).reshape(1, -1)
        X = QUALITY_SCALER.transform(features)
        pred = QUALITY_MODEL.predict(X)[0]
        proba = QUALITY_MODEL.predict_proba(X)[0]
        label = "good" if pred == 1 else "bad"
        confidence = float(max(proba))
        return jsonify({"label": label, "confidence": round(confidence, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/quality', methods=['POST'])
def quality():
    try:
        body = request.get_json()
        samples = body.get("samples", [])
        if len(samples) < 50:
            return jsonify({"quality": "bad", "confidence": 0}), 200
        import statistics
        mean = statistics.mean(samples)
        var = statistics.variance(samples) if len(samples) > 1 else 0
        r = max(samples) - min(samples) if samples else 0
        if var < 1 and r < 5:
            return jsonify({"quality": "bad", "confidence": 0.5}), 200
        if r > 50 and var > 10:
            return jsonify({"quality": "excellent", "confidence": 0.8}), 200
        return jsonify({"quality": "acceptable", "confidence": 0.7}), 200
    except Exception as e:
        return jsonify({"quality": "bad", "confidence": 0, "error": str(e)}), 200
    
# 在 app.py 中添加这个函数和路由

def find_valleys(samples, fps=30):
    """
    谷底检测算法（从 TypeScript 的 ppg.ts 移植过来）
    返回谷底索引列表
    """
    if len(samples) < 3:
        return []
    
    valleys = []
    # 寻找局部最小值（谷底）
    for i in range(1, len(samples) - 1):
        if samples[i] <= samples[i-1] and samples[i] <= samples[i+1]:
            valleys.append(i)
    
    # 可选：添加最小距离过滤（避免太近的谷底）
    min_distance = int(fps * 0.3)  # 至少间隔 0.3 秒
    filtered = []
    for v in valleys:
        if not filtered or (v - filtered[-1]) >= min_distance:
            filtered.append(v)
    
    return filtered


def compute_heart_rate_from_valleys(valleys, fps=30):
    """
    从谷底间隔计算心率和置信度
    """
    if len(valleys) < 2:
        return {"bpm": 0, "confidence": 0}
    
    # 计算间隔（采样点数）
    intervals = []
    for i in range(1, len(valleys)):
        interval_samples = valleys[i] - valleys[i-1]
        intervals.append(interval_samples)
    
    # 转换为 BPM
    bpm_list = [(60 * fps) / interval for interval in intervals]
    
    if not bpm_list:
        return {"bpm": 0, "confidence": 0}
    
    # 计算平均 BPM
    avg_bpm = np.mean(bpm_list)
    
    # 计算置信度（基于变异系数）
    if avg_bpm > 0:
        cv = np.std(bpm_list) / avg_bpm
        confidence = max(0, min(1, 1 - cv))
    else:
        confidence = 0
    
    return {"bpm": round(avg_bpm, 1), "confidence": round(confidence, 2)}


def compute_hrv_from_valleys(valleys, fps=30):
    """
    从谷底间隔计算心率变异性 (SDNN)
    """
    if len(valleys) < 3:
        return {"sdnn": 0, "confidence": 0}
    
    # 计算间隔（毫秒）
    intervals_ms = []
    for i in range(1, len(valleys)):
        interval_seconds = (valleys[i] - valleys[i-1]) / fps
        intervals_ms.append(interval_seconds * 1000)
    
    if len(intervals_ms) < 2:
        return {"sdnn": 0, "confidence": 0}
    
    # 计算 SDNN（标准差）
    sdnn = np.std(intervals_ms)
    
    # 置信度基于可用间隔数量
    confidence = min(1.0, len(intervals_ms) / 10)
    
    return {"sdnn": round(sdnn, 1), "confidence": round(confidence, 2)}


@app.route('/analyze', methods=['POST'])
def analyze_ppg():
    """
    服务器端 PPG 分析
    接收: { "samples": number[], "fps": number }
    返回: { "valleys": number[], "heartRate": {bpm, confidence}, "hrv": {sdnn, confidence} }
    """
    try:
        body = request.get_json()
        if not body:
            return jsonify({"error": "No body"}), 400
        
        samples = body.get("samples", [])
        fps = body.get("fps", 30)
        
        if not isinstance(samples, list) or len(samples) < MIN_SAMPLES_FOR_DETECTION:
            return jsonify({
                "error": f"samples must be an array with at least {MIN_SAMPLES_FOR_DETECTION} points"
            }), 400
        
        # 执行谷底检测
        valleys = find_valleys(samples, fps)
        
        # 计算心率
        heart_rate = compute_heart_rate_from_valleys(valleys, fps)
        
        # 计算 HRV
        hrv = compute_hrv_from_valleys(valleys, fps)
        
        return jsonify({
            "valleys": valleys,
            "heartRate": heart_rate,
            "hrv": hrv
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/upload-model', methods=['POST'])
def upload_model():
    """Receive base64 encoded model files and load them into memory"""
    try:
        body = request.get_json()
        if not body or 'model' not in body or 'scaler' not in body:
            return jsonify({"success": False, "error": "Missing model or scaler"}), 400
        
        global QUALITY_MODEL, QUALITY_SCALER
        
        # Decode base64 and load models
        QUALITY_MODEL = joblib.load(io.BytesIO(base64.b64decode(body['model'])))
        QUALITY_SCALER = joblib.load(io.BytesIO(base64.b64decode(body['scaler'])))
        
        return jsonify({"success": True, "message": "Model uploaded"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500