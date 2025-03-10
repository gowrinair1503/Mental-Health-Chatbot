import streamlit as st
import torch
import requests
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load model from Google Drive
MODEL_URL = "https://drive.google.com/file/d/12_9u-6oSlhMa491-yS48DaEl2HMtxct3/view?usp=drive_link"  # Replace with your actual Google Drive link

@st.cache_resource()
def load_model():
    # Download the model file
    model_path = "model.safetensors"
    with open(model_path, "wb") as f:
        f.write(requests.get(MODEL_URL).content)
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

# Load the model
tokenizer, model = load_model()

# Function to generate a response
def generate_response(user_input):
    input_text = user_input + " [SEP]"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape)
    output = model.generate(
        input_ids, attention_mask=attention_mask, max_length=100,
        pad_token_id=tokenizer.eos_token_id, do_sample=True,
        top_k=50, temperature=0.7
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit UI - Pastel Theme
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💙", layout="centered")

# Custom CSS for pastel colors and chat format
st.markdown("""
    <style>
        body {
            background-color: #f7f3e9;
            color: #333;
        }
        .stTextInput>div>div>input {
            border: 2px solid #a3c9c7;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #a3c9c7;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 8px 15px;
        }
        .stButton>button:hover {
            background-color: #87b5b0;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧘 Mental Health Chatbot")
st.write("A calming chatbot to support your mental health. 💙")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input field
user_input = st.chat_input("Type your message...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate bot response
    response = generate_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display bot message
    with st.chat_message("assistant"):
        st.write(response)

