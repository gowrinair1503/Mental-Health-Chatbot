import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define your Hugging Face model repo
HF_MODEL_REPO = "your-username/your-model-repo"  # Replace with your actual repo name

# Load model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
model = AutoModelForCausalLM.from_pretrained(HF_MODEL_REPO, torch_dtype=torch.float16)

def generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    output = model.generate(**inputs, max_length=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit App
st.title("Mental Health Companion Chatbot")
st.write("A chatbot to support mental well-being. Type a message below.")

user_input = st.text_input("You:", "")

if user_input:
    response = generate_response(user_input)
    st.text_area("Chatbot:", response, height=100)
