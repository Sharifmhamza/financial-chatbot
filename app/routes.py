from flask import Flask, request, render_template, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import json
import os

app = Flask(__name__, template_folder='../templates')

MODEL_PATH = "model"
pipe = pipeline("fill-mask", model="google-bert/bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")
model.eval()

with open("data/intents.json", "r") as f:
    intents = json.load(f)
with open("data/response_templates.json", "r") as f:
    templates = json.load(f)

def fill_slots(intent_name, user_input):
    if "ETF" in user_input.upper():
        return templates[intent_name].replace("{product}", "ETFs like VTI and AGG")
    elif "stock" in user_input.lower():
        return templates[intent_name].replace("{product}", "blue-chip stocks")
    else:
        return templates[intent_name].replace("{product}", "diversified mutual funds")

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=1).item()
    return intents[pred]["intent"]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["message"]
    intent_name = predict_intent(user_input)
    response = fill_slots(intent_name, user_input)
    return jsonify({"response": response})

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    log_path = "feedback/feedback_log.json"
    if not os.path.exists("feedback"):
        os.makedirs("feedback")
    with open(log_path, "a") as f:
        f.write(json.dumps(data) + "\n")
    return jsonify({"status": "logged"})
