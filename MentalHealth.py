import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Define the model location (Update with your actual Hugging Face model path)
MODEL_NAME = "grpnair2003/Mental-Health-Chatbot/blob/main/model.safetensors"

@st.cache_resource()
def load_model():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load the model
tokenizer, model = load_model()

def generate_response(user_input):
    """Generate chatbot response based on user input."""
    if not tokenizer or not model:
        return "Model failed to load. Please check your settings."

    input_text = user_input + " [SEP]"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape)

    output = model.generate(
        input_ids, attention_mask=attention_mask, max_length=100,
        pad_token_id=tokenizer.eos_token_id, do_sample=True,
        top_k=50, temperature=0.7
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Chatbot")
st.write("I'm here to help. How do you feel today?")

# Input box
user_input = st.text_input("You:", "")

# Generate response when user clicks "Send"
if st.button("Send"):
    if user_input.strip():
        response = generate_response(user_input)
        st.write("🤖 **Bot:**", response)
    else:
        st.warning("Please enter a message.")

# Additional features
st.sidebar.header("Settings")
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        body { background-color: #121212; color: white; }
        .stTextInput, .stButton { color: white !important; }
        </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown("### Resources")
st.sidebar.write("[🧘 Meditation Guide](https://www.headspace.com/)")
st.sidebar.write("[📞 Helplines](https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response)")

st.sidebar.markdown("---")

