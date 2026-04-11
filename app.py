import json
import os
import numpy as np
import torch
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)


MODEL_DIR = Path("model_output/best_model")
META_PATH = Path("model_output/training_meta.json")
MAX_LEN   = 96
MODEL     = None
TOKENIZER = None
META      = {}
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the trained model and metadata. Called once at app startup."""
    global MODEL, TOKENIZER, META

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"\n❌  Model not found at '{MODEL_DIR}'.\n"
            f"    Run  python train.py  first to train and save the model.\n"
        )

    print(f"🤖 Loading model from {MODEL_DIR} ...")
    TOKENIZER = DistilBertTokenizer.from_pretrained(str(MODEL_DIR))
    MODEL     = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"   Device: {DEVICE}")

    if META_PATH.exists():
        with open(META_PATH) as f:
            META = json.load(f)
        print(f"   Trained: {META.get('trained_at','?')}")
        print(f"   Best val acc: {META.get('best_val_acc', 0)*100:.2f}%")
    else:
        META = {}

    print("✅ Model ready.\n")


def row_to_text(amount, txn_type, category, country, hour,
                device_risk, ip_risk):
    """Must match exactly what train.py used."""
    hour = int(hour)
    if hour < 5:       time_desc = "late night"
    elif hour < 12:    time_desc = "morning"
    elif hour < 17:    time_desc = "afternoon"
    elif hour < 21:    time_desc = "evening"
    else:              time_desc = "night"

    device_risk = float(device_risk)
    ip_risk     = float(ip_risk)
    dev_desc = "high risk" if device_risk > 0.7 else ("medium risk" if device_risk > 0.4 else "low risk")
    ip_desc  = "high risk" if ip_risk     > 0.7 else ("medium risk" if ip_risk     > 0.4 else "low risk")

    return (
        f"amount: {round(float(amount), 2)} | "
        f"type: {str(txn_type).lower()} | "
        f"category: {str(category).lower()} | "
        f"country: {str(country).lower()} | "
        f"hour: {hour} ({time_desc}) | "
        f"device risk: {dev_desc} ({round(device_risk, 3)}) | "
        f"ip risk: {ip_desc} ({round(ip_risk, 3)})"
    )


def risk_level(fraud_prob):
    if fraud_prob < 0.30: return "LOW"
    if fraud_prob < 0.55: return "MEDIUM"
    if fraud_prob < 0.80: return "HIGH"
    return "CRITICAL"


@torch.no_grad()
def predict(amount, txn_type, category, country, hour, device_risk, ip_risk):
    text = row_to_text(amount, txn_type, category, country, hour,
                    device_risk, ip_risk)
    enc = TOKENIZER(
        text, max_length=MAX_LEN, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    logits = MODEL(
        input_ids=enc["input_ids"].to(DEVICE),
        attention_mask=enc["attention_mask"].to(DEVICE),
    ).logits
    probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
    fraud_prob = float(probs[1])
    label      = 1 if fraud_prob >= 0.5 else 0
    confidence = fraud_prob if label == 1 else 1 - fraud_prob

    # Build explanation flags
    flags = []
    if float(amount) > 5000:   flags.append(f"High amount (₹{float(amount):,.0f})")
    if int(hour) in {0,1,2,3,22,23}: flags.append(f"Suspicious hour ({hour}:00)")
    if str(txn_type) in {"ATM","Online"}: flags.append(f"High-risk type ({txn_type})")
    if float(device_risk) > 0.7: flags.append(f"High device risk ({device_risk:.2f})")
    if float(ip_risk)     > 0.7: flags.append(f"High IP risk ({ip_risk:.2f})")
    if not flags: flags.append("No strong individual risk signals detected")

    return {
        "label":           label,
        "fraud_probability": round(fraud_prob, 4),
        "confidence":      round(confidence, 4),
        "risk_level":      risk_level(fraud_prob),
        "flags":           flags,
        "text_input":      text,
    }



@app.route("/")
def index():
    return render_template("index.html", meta=META)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    try:
        result = predict(
            amount      = data["amount"],
            txn_type    = data["transaction_type"],
            category    = data["merchant_category"],
            country     = data["country"],
            hour        = data["hour"],
            device_risk = data["device_risk_score"],
            ip_risk     = data["ip_risk_score"],
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/model_info")
def api_model_info():
    return jsonify({
        "loaded":      MODEL is not None,
        "model_dir":   str(MODEL_DIR),
        "device":      str(DEVICE),
        "meta":        META,
    })


if __name__ == "__main__":
    load_model()
    print("🌐 Starting server at http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
