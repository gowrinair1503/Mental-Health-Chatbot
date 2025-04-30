from flask import Flask, render_template, request, jsonify, url_for
import torch
import pickle
import random
import json
import numpy as np
from safetensors.torch import load_file
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# === Load model, tokenizer, encoder, responses ===
def load_all():
    tokenizer = DistilBertTokenizerFast.from_pretrained("./")
    model = DistilBertForSequenceClassification.from_pretrained("./", config="config.json", ignore_mismatched_sizes=True)
    state_dict = load_file("model.safetensors")
    model.load_state_dict(state_dict)
    model.eval()

    # Instead of loading a pickle file, load JSON
    with open("label_encoder.json", "r") as f:
        label_data = json.load(f)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array([label_data[str(i)] for i in range(len(label_data))])

    with open("Updated_MentalHealthChatbotDataset_with_overthinking_panic.json") as f:
        data = json.load(f)
        response_map = {intent["tag"]: intent["responses"] for intent in data["intents"]}

    return tokenizer, model, label_encoder, response_map

tokenizer, model, le, response_map = load_all()

SUGGESTIONS = [
    " Try a 5-minute deep breathing session or write down 3 things you're grateful for.",
    " You can also play calming music or journal your thoughts.",
    " Take a walk outside in nature and focus on the present moment.",
    " Take a warm bath to relax your body and mind.",
    " Try meditation or mindfulness exercises to ease stress.",
    " Step outside for some fresh air and let the sunshine in!",
    " Read a book or listen to an audiobook to escape into another world.",
    " Try some creative activities like drawing, painting, or coloring.",
    " Go for a quick walk or do light exercise to boost your energy.",
    " Apply your favorite lotion or essential oils to help calm your mind.",
    " Practice gratitude by writing down 3 things you're thankful for today.",
    " Take a break from screens and do something that relaxes you.",
    " Take a short nap if you need to rest your mind and body.",
    " Enjoy a warm cup of tea, coffee, or hot chocolate.",
    " Try a relaxing bedtime routine to help you unwind for sleep.",
    " Reach out to a friend or family member for a chat to feel connected.",
    " Do some light stretching to relieve tension in your muscles.",
    " If needed, talk to a professional about how you're feeling.",
    " Give yourself some positive affirmations: 'I am worthy of love and peace.'",
    " Listen to your favorite calming playlist or nature sounds.",
    " Plan something fun to look forward to in the coming days.",
    " Do something kind for yourself today, even if it's something small.",
    " Celebrate small wins! No matter how small, progress is still progress.",
    " Focus on one task at a time and celebrate your achievements.",
    " Practice positive self-talk, reminding yourself you're doing your best.",
    " Practice deep breathing exercises: inhale for 4 counts, hold for 7, exhale for 8.",
    " Take a break by visiting a local park and observing nature around you.",
    " Engage in a hobby you enjoy, such as knitting, writing, or gardening.",
    " Try progressive muscle relaxation to help release physical tension.",
    " Engage in yoga to connect your mind and body and release tension.",
    " Take a moment to just sit quietly and focus on your breath."
]

def get_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    tag = le.inverse_transform([pred_id])[0]
    response = random.choice(response_map.get(tag, ["I'm here for you."]))
    return tag, response

chat_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history
    message = ""
    suggestion = random.choice(SUGGESTIONS)
    if request.method == "POST":
        user_input = request.form.get("message", "").strip()
        if user_input:
            tag, bot_reply = get_response(user_input)
            chat_history.append(("user", user_input))
            chat_history.append(("yara", bot_reply))

            if tag in ["sad", "stressed", "worthless"]:
                suggestion = random.choice(SUGGESTIONS)

    return render_template("index.html", chat_history=chat_history, suggestion=suggestion)

@app.route("/refresh_suggestion", methods=["GET"])
def refresh_suggestion():
    new_suggestion = random.choice(SUGGESTIONS)
    return jsonify({"new_suggestion": new_suggestion})

@app.route("/new_chat", methods=["POST"])
def new_chat():
    global chat_history
    chat_history.clear()  # Clear it
    return jsonify({"redirect": url_for('index')})

if __name__ == "__main__":
    app.run(debug=True)
