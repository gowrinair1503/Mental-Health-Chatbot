import streamlit as st
import os
import gdown
import zipfile
import json
import random
import torch
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_PATH = "mental_health_chatbot_model"
ENCODER_PATH = "label_encoder.json"

@st.cache_resource
def download_and_load():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        file_id = "1c_JsZ5PUZhbrg_zDeNotoVlXQPz7zif8"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "model.zip"
        gdown.download(url, output, quiet=False)

        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall()  

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

    encoder_file_path = os.path.join(MODEL_PATH, ENCODER_PATH)
    with open(encoder_file_path, "r") as f:
        label_mapping = json.load(f)
    id_to_label = {int(k): v for k, v in label_mapping.items()}

    return model, tokenizer, id_to_label

model, tokenizer, id_to_label = download_and_load()


response_map = {
    "greeting": ["Hello! I'm here to support your mental well-being."],
    "goodbye": ["Take care! You're not alone."],
    "scared": ["It's okay to feel scared. Want to talk about it?"],
}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    tag = id_to_label[pred_id]
    response = random.choice(response_map.get(tag, ["I'm here for you. Tell me more."]))
    return tag, response

# Streamlit UI
st.set_page_config(page_title="Yara", page_icon="🧘", layout="centered")
st.title("Yara")
st.write("Type in how you're feeling. I'm here to support you.")

user_input = st.text_input("You:", placeholder="I'm feeling anxious...")

if st.button("Send") and user_input.strip():
    tag, reply = predict(user_input)
    st.success(f"**Yara ({tag})**: {reply}")
