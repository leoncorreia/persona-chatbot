import streamlit as st
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel, configure
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI
from elevenlabs import ElevenLabs

# --- CONFIGURATION ---
MODEL_MAPPING = {
    "Gemini 2.5 Flash": {"provider": "google", "model": "gemini-2.5-flash"},
    "GPT-4.0": {"provider": "openai", "model": "gpt-4o-mini"},
    "Llama 3.1 8B": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    "Claude": {"provider": "groq", "model": "mixtral-8x7b-32768"},
}

PERSONA_MAPPING = {
    "Elon Musk": "elon",
    "David Attenborough": "david_attenborough",
    "Morgan Freeman": "morgan_freeman"
}

PERSONA_INFO = {
    "Elon Musk": {"emoji": "🚀", "tagline": "First Principles", "color": "#1DA1F2", "avatar": "🚀"},
    "David Attenborough": {"emoji": "🌍", "tagline": "Nature's Voice", "color": "#2E7D32", "avatar": "🌿"},
    "Morgan Freeman": {"emoji": "✨", "tagline": "Cosmic Wisdom", "color": "#6A1B9A", "avatar": "🎙️"}
}

# --- PAGE CONFIG ---
st.set_page_config(page_title="Persona Chat", page_icon="🎭", layout="wide")

# --- CUSTOM CSS (ChatGPT/Gemini Style) ---
st.markdown("""
<style>
    /* 1. FORCE WHITE THEME & REMOVE PADDING */
    .stApp { background-color: #ffffff; color: #333333; }
    .block-container { padding-top: 1rem !important; padding-bottom: 5rem !important; max-width: 50rem !important; }
    header { visibility: hidden; } /* Hide Streamlit default header */
    
    /* 2. CHAT INPUT STYLING (Floating Bottom) */
    .stChatInput {
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        width: 100%; 
        max-width: 48rem; /* Match ChatGPT width */
        z-index: 100;
    }
    
    /* 3. DROPDOWN STYLING (Minimalist) */
    div[data-testid="stSelectbox"] {
        border: none;
        background-color: transparent;
    }
    div[data-testid="stSelectbox"] > div > div {
        background-color: #f7f7f8; /* Light grey pill */
        border: none;
        border-radius: 8px;
        color: #333;
        font-weight: 500;
    }
    
    /* 4. PERSONA BUTTONS (Centered Hero) */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid #e5e5e5;
        background-color: white;
        color: #333;
        padding: 1rem;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton > button:hover {
        border-color: #8B5CF6;
        color: #8B5CF6;
        background-color: #fcfaff;
        transform: translateY(-2px);
    }
    
    /* 5. MESSAGE BUBBLES */
    .stChatMessage { background-color: transparent; border: none; }
    div[data-testid="stChatMessageContent"] {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
if "messages" not in st.session_state: st.session_state.messages = []
if "selected_persona" not in st.session_state: st.session_state.selected_persona = "David Attenborough"
if "selected_model" not in st.session_state: st.session_state.selected_model = list(MODEL_MAPPING.keys())[0]

# --- BACKEND MOCK (Replace with your actual logic) ---
def get_llm_response(prompt, model, persona):
    return f"This is a simulated answer from **{persona}** using **{model}**. (Connect API to make this real!)"

# --- UI LAYOUT ---

# 1. TOP BAR (Model Selector & Persona Indicator)
# We use columns to put the model selector top-left (Gemini style)
top_col1, top_col2 = st.columns([1, 4])

with top_col1:
    # Model Dropdown (The "Gemini" switcher)
    st.session_state.selected_model = st.selectbox(
        "Model", 
        list(MODEL_MAPPING.keys()), 
        index=0, 
        label_visibility="collapsed"
    )

with top_col2:
    # Subtle indicator of who we are talking to
    curr_info = PERSONA_INFO[st.session_state.selected_persona]
    st.markdown(f"<div style='text-align: right; color: #888; padding-top: 5px; font-size: 0.9rem;'>Talking to <b>{st.session_state.selected_persona}</b> {curr_info['emoji']}</div>", unsafe_allow_html=True)

st.markdown("---") # Minimal divider

# 2. MAIN CONTENT AREA

# SCENARIO A: Chat is Empty -> Show "Hero" / Welcome Screen
if len(st.session_state.messages) == 0:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; font-size: 2.5rem; color: #333;'>How can {st.session_state.selected_persona} help?</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #666; margin-bottom: 3rem;'>{PERSONA_INFO[st.session_state.selected_persona]['tagline']}</p>", unsafe_allow_html=True)
    
    # Centered Persona Selection Grid
    c1, c2, c3 = st.columns(3)
    
    def set_persona(p):
        st.session_state.selected_persona = p
        
    for idx, (name, info) in enumerate(PERSONA_INFO.items()):
        # We put buttons in columns to center them
        col = [c1, c2, c3][idx]
        with col:
            # If this button is clicked, it updates the state and reruns
            if st.button(f"{info['emoji']} {name}", use_container_width=True):
                st.session_state.selected_persona = name
                st.rerun()

# SCENARIO B: Chat Active -> Show History
else:
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else PERSONA_INFO[st.session_state.selected_persona]["avatar"]
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# 3. CHAT INPUT (Pinned to Bottom)
if prompt := st.chat_input("Ask a question..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Rerun immediately to update UI (switch from Hero to Chat view)
    st.rerun()

# 4. HANDLE RESPONSE (If last message is user)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant", avatar=PERSONA_INFO[st.session_state.selected_persona]["avatar"]):
        with st.spinner("Thinking..."):
            response = get_llm_response(
                st.session_state.messages[-1]["content"], 
                st.session_state.selected_model, 
                st.session_state.selected_persona
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})