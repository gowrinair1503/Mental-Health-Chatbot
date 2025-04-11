import streamlit as st
import os
import gdown
import zipfile
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import joblib
import random

MODEL_PATH = "mental_health_chatbot_model"
ENCODER_PATH = "label_encoder.pkl"

@st.cache_resource
def download_and_load():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        file_id = "1c_JsZ5PUZhbrg_zDeNotoVlXQPz7zif8"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "model.zip"
        gdown.download(url, output, quiet=False)

        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(MODEL_PATH)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    return model, tokenizer

model, tokenizer = download_and_load()
le = joblib.load(ENCODER_PATH)

# Basic response map (expand as needed)
response_map = {
    "greeting": ["Hello! I'm here to support your mental well-being."],
    "goodbye": ["Take care! You're not alone."],
    "scared": ["It's okay to feel scared. Want to talk about it?"],
    # Add more tags as needed
}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    tag = le.inverse_transform([pred_id])[0]
    response = random.choice(response_map.get(tag, ["I'm here for you. Tell me more."]))
    return tag, response

# Streamlit UI
st.set_page_config(page_title="Yara", page_icon="🧘", layout="centered")
st.title("Yara")
st.write("Type in how you're feeling. I'm here to support you.")

user_input = st.text_input("You:", placeholder="I'm feeling anxious...")

if st.button("Send") and user_input.strip():
    tag, reply = predict(user_input)
    st.success(f"**Pandora ({tag})**: {reply}")
