import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load model and tokenizer
MODEL_PATH = "model.safetensors"  # Update this to the correct path
@st.cache_resource()
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def generate_response(user_input):
    input_text = user_input + " [SEP]"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape)
    output = model.generate(
        input_ids, attention_mask=attention_mask, max_length=100,
        pad_token_id=tokenizer.eos_token_id, do_sample=True,
        top_k=50, temperature=0.7
    )
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Streamlit UI
st.set_page_config(page_title="Mental Health Chatbot", layout="centered", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    body {background-color: #FAF3E0; color: #333;}
    .stApp {background-color: #FAF3E0;}
    .chat-container {border-radius: 10px; padding: 10px;}
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("🌙 Theme")
mode = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
if mode == "Dark":
    st.markdown("""
    <style>
    body {background-color: #2C2C2C; color: #FFF;}
    .stApp {background-color: #2C2C2C;}
    </style>
    """, unsafe_allow_html=True)

st.title("🧘 Mental Health Chatbot")
st.write("Your AI companion for mental well-being.")

# Chat format
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.text_input("You:", "", key="user_input")
if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = generate_response(user_input)
    st.session_state.messages.append({"role": "bot", "content": response})
    st.rerun()
