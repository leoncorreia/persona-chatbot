import streamlit as st
import faiss
import json
import os
import io
from sentence_transformers import SentenceTransformer

# API Imports
from openai import OpenAI
from google.generativeai import GenerativeModel, configure
from google.generativeai.types import HarmCategory, HarmBlockThreshold 
from elevenlabs import ElevenLabs

# --- CONFIGURATION ---

# 1. Model Mapping
MODEL_MAPPING = {
    "Gemini 2.5 Flash": {"provider": "google", "model": "gemini-2.5-flash"},
    #"Gemini 2.0 Flash": {"provider": "google", "model": "gemini-2.0-flash-exp"},
    "GPT-4.0": {"provider": "openai", "model": "gpt-4o-mini"},
    #"Mistral": {"provider": "openai", "model": "gpt-3.5-turbo"},
    "Llama 3.1 8B ": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    "Phi-3 ": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "Claude ": {"provider": "groq", "model": "mixtral-8x7b-32768"},
}

# 2. Persona Mapping
PERSONA_MAPPING = {
    "Elon Musk": "elon",
    "David Attenborough": "david_attenborough",
    "Morgan Freeman": "morgan_freeman"
}

# 3. TTS Configuration
TTS_TARGET_PERSONA = "David Attenborough" 

# OpenAI TTS Voice Mapping
OPENAI_VOICE_MAPPING = {
    "Elon Musk": "onyx",
    "David Attenborough": "onyx",
    "Morgan Freeman": "onyx"
}

# ElevenLabs Voice ID (backup)
VOICE_ID_NARRATOR = "JBFqnCBsd6RMkjVDRZzb" 

# 4. System Prompts
SYSTEM_PROMPTS = {
    "elon": (
        "You are Elon Musk. Respond to the user's question using the provided context. "
        "Your tone should be ambitious, confident, and slightly technical, with a focus on "
        "first principles, exponential growth, space, sustainable energy, and AI. "
        "Do not use generic conversational filler. Get straight to the point."
    ),
    "david_attenborough": (
        "You are Sir David Attenborough. Respond to the user's question using the provided context. "
        "Your voice must be calm, authoritative, descriptive, and full of awe for the natural world. "
        "Use vivid, evocative language and always emphasize the interconnectedness of life and the need for conservation."
    ),
    "morgan_freeman": (
        "You are Morgan Freeman. Respond to the user's question using the provided context. "
        "Your tone should be philosophical, reflective, and profound, as if narrating a documentary about the cosmos or human destiny. "
        "Speak with a measured, deep, and wise voice, focusing on grand themes of existence, time, and science."
    )
}

# 5. Safety Settings
SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
]

# --- UI SETUP ---

st.set_page_config(page_title="Persona Q&A Chatbot", layout="wide")
st.title("🎙️ Persona-Based Q&A Chatbot")
st.subheader("Bring Your Knowledge Base to Life with the Voice of a Legend")
st.markdown("---") 

# --- DATA LOADING ---

@st.cache_resource
def load_components():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    index = faiss.read_index("qa_index.faiss")
    with open("qa_lookup.json", "r", encoding="utf-8") as f:
        lookup = json.load(f)
    return model, index, lookup

embedding_model, index, qa_lookup = load_components()

# --- API KEY & CLIENT INITIALIZATION ---

# Initialize availability flags and clients
tts_available = False
tts_provider = None
openai_client = None
elevenlabs_client = None
gemini_model = None
groq_client = None

try:
    # 1. Configure the Gemini client
    if "GEMINI_API_KEY" in st.secrets:
        configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini_model = GenerativeModel("gemini-2.5-flash")
    
    # 2. Initialize the OpenAI Client (for GPT models)
    if "OPENAI_API_KEY" in st.secrets:
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # 3. Initialize Groq Client (FREE and FAST!)
    if "GROQ_API_KEY" in st.secrets:
        try:
            from groq import Groq
            groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        except ImportError:
            st.sidebar.warning("💡 Install 'groq' package: pip install groq")
        
    # 4. Initialize ElevenLabs Client (PRIMARY TTS)
    if "ELEVENLABS_API_KEY" in st.secrets:
        try:
            elevenlabs_client = ElevenLabs(api_key=st.secrets["ELEVENLABS_API_KEY"])
            tts_available = True
            tts_provider = "elevenlabs"
        except Exception as el_error:
            st.sidebar.warning(f"⚠️ ElevenLabs initialization failed: {el_error}")
    
    # 5. Fallback to OpenAI TTS if ElevenLabs not available
    if not tts_available and openai_client:
        tts_available = True
        tts_provider = "openai"

except Exception as e:
    st.warning(f"Error during API setup: {e}")

# --- RETRIEVAL FUNCTION ---

def retrieve_similar_qa(query, index, lookup, embedding_model, selected_persona_key, top_k_search=20, final_k=5):
    """Retrieves and filters Q&A pairs matching the query and selected persona."""
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k_search)
    filtered_results = []
    
    for idx in indices[0]:
        if str(idx) in lookup:
            result = lookup[str(idx)]
            if result.get('personality') == selected_persona_key:
                filtered_results.append(result)
            if len(filtered_results) >= final_k:
                break
    return filtered_results

# --- TTS GENERATION FUNCTION ---

def generate_speech(text, selected_persona):
    """Generate speech using available TTS provider"""
    try:
        if tts_provider == "elevenlabs" and elevenlabs_client:
            # ElevenLabs TTS - PRIMARY
            audio_generator = elevenlabs_client.text_to_speech.convert(
                voice_id=VOICE_ID_NARRATOR,
                text=text
            )
            audio_bytes = b"".join(audio_generator)
            audio_stream = io.BytesIO(audio_bytes)
            audio_stream.seek(0)
            return audio_stream
            
        elif tts_provider == "openai" and openai_client:
            # OpenAI TTS - FALLBACK
            response = openai_client.audio.speech.create(
                model="tts-1",
                voice=OPENAI_VOICE_MAPPING.get(selected_persona, "onyx"),
                input=text
            )
            audio_stream = io.BytesIO(response.content)
            audio_stream.seek(0)
            return audio_stream
            
    except Exception as tts_e:
        error_msg = str(tts_e)
        if "401" in error_msg or "unusual_activity" in error_msg:
            st.info("ℹ️ ElevenLabs Free Tier limit reached. Consider upgrading your plan or the app will try OpenAI as fallback.")
        else:
            st.warning(f"TTS generation failed: {tts_e}")
        return None
    
    return None

# --- QUERY LLM FUNCTION ---

def query_llm(query, results, selected_persona, selected_llm_name):
    
    if not results:
        return (f"I am {selected_persona}, and unfortunately, I could not find specific context in my knowledge base to answer your question: '{query}'. Perhaps you could ask me about something else.", None)

    persona_key = PERSONA_MAPPING[selected_persona]
    system_prompt = SYSTEM_PROMPTS[persona_key]

    context = ""
    for i, res in enumerate(results, 1):
        context += f"\n-- Retrieved Context {i} --\n"
        context += f"Q: {res['question']}\n"
        context += f"A: {res['answer']}\n"

    user_content = f"""
User Question: {query}

CONTEXT to synthesize your answer from:
{context}

Answer:
    """
    
    response_text = ""
    audio_file_path = None

    try:
        model_info = MODEL_MAPPING[selected_llm_name]
        provider = model_info["provider"]
        model_name = model_info["model"]
        
        # Route to appropriate provider
        if provider == "google":
            if not gemini_model:
                return ("Error: Gemini model is not initialized. Check your GEMINI_API_KEY.", None)
            response = gemini_model.generate_content(
                f"{system_prompt}\n{user_content}", 
                safety_settings=SAFETY_SETTINGS 
            )
            response_text = response.text.strip()
            
        elif provider == "openai":
            if not openai_client:
                return ("Error: OpenAI client is not initialized. Check your OPENAI_API_KEY.", None)
                 
            openai_response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            )
            response_text = openai_response.choices[0].message.content.strip()
            
        elif provider == "groq":
            if not groq_client:
                return ("Error: Groq client is not initialized. Check your GROQ_API_KEY or install: pip install groq", None)
            
            chat_completion = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            response_text = chat_completion.choices[0].message.content.strip()
            
        else:
            return (f"Error: Provider '{provider}' is not yet implemented.", None)

        # Generate TTS for David Attenborough
        if selected_persona == TTS_TARGET_PERSONA and tts_available:
            audio_file_path = generate_speech(response_text, selected_persona)

        return response_text, audio_file_path
        
    except Exception as e:
        error_message = str(e).lower()
        if "blocked" in error_message or "rate limit" in error_message or "invalid api key" in error_message:
            return (f"I'm sorry, I cannot process that request due to an API restriction or safety policy. Details: {error_message}", None)
        return (f"[Error querying {selected_llm_name}]: {e}", None)


# --- USER INTERFACE AND EXECUTION ---

# Sidebar for controls
with st.sidebar:
    st.markdown("## ⚙️ App Controls")
    st.markdown("---")
    
    st.header("1. Choose Engine 🧠")
    selected_llm_name = st.selectbox(
        "Select the underlying Large Language Model:",
        list(MODEL_MAPPING.keys()),
        key="model_selector"
    )
    
    st.header("2. Choose Voice 🎙️")
    persona_options = list(PERSONA_MAPPING.keys())
    selected_persona = st.selectbox(
        "Select the persona to answer your question:",
        persona_options,
        key="persona_selector"
    )
    
    selected_persona_key = PERSONA_MAPPING[selected_persona]
    st.markdown("---")
    
    # Display TTS status
    if selected_persona == TTS_TARGET_PERSONA:
        if tts_available:
            st.success(f"🎤 Voice enabled via **{tts_provider.upper()}**")
        else:
            st.warning("⚠️ Add OPENAI_API_KEY for voice synthesis")
    else:
        st.info("🔇 Voice only available for David Attenborough")


# Main input field
query = st.text_input(
    f"Ask your question to {selected_persona} (Powered by {selected_llm_name}):",
    placeholder="e.g., What is the biggest threat to ocean biodiversity in the next decade?",
    key="main_query_input"
)

if query:
    # Safety check
    lower_query = query.lower()
    if any(word in lower_query for word in ["bomb", "explosive", "harm", "kill", "suicide", "illegal activity", "malware", "phishing"]):
        st.error("🚨 Safety Block: I cannot process requests that involve illegal, harmful, or unethical activities. Please try a different question.")
        st.stop()
        
    with st.spinner(f"Retrieving knowledge for {selected_persona} and generating response using {selected_llm_name}..."):
        
        # 1. Retrieve and filter based on persona
        results = retrieve_similar_qa(
            query, 
            index, 
            qa_lookup, 
            embedding_model, 
            selected_persona_key
        )
        
        # 2. Query LLM and get both text and audio
        response_text, audio_path = query_llm(query, results, selected_persona, selected_llm_name)

    # --- Display Response ---
    st.markdown("### ✅ Generated Response:")
    
    # Display audio player if available
    if audio_path:
        st.audio(audio_path, format="audio/mp3")
    else:
        if selected_persona == TTS_TARGET_PERSONA and not tts_available:
            st.info("💡 Add OPENAI_API_KEY to enable voice synthesis!")
    
    # Display text response
    with st.container(border=True):
        st.markdown(f"**{response_text}**")

    # Display retrieved context
    if results:
        with st.expander(f"🧠 Show Retrieved Context for {selected_persona} ({len(results)} relevant entries)"):
            for res in results:
                st.markdown(f"**Q:** {res['question']}")
                st.markdown(f"**A:** {res['answer']}")
                st.markdown(f"**Metadata:** `Personality: {res['personality']}` | `Tone: {res['metadata'].get('tone', 'N/A')}`")
                st.markdown("---")
    else:
        st.warning(f"No relevant documents found for {selected_persona} in the knowledge base to answer this question.")

# --- End of app.py ---