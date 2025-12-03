# import streamlit as st
# import faiss
# import json
# import os
# import io
# from sentence_transformers import SentenceTransformer

# # API Imports
# from openai import OpenAI
# from google.generativeai import GenerativeModel, configure
# from google.generativeai.types import HarmCategory, HarmBlockThreshold 
# from elevenlabs import ElevenLabs

# # --- CONFIGURATION ---

# # 1. Model Mapping
# MODEL_MAPPING = {
#     "Gemini 2.5 Flash": {"provider": "google", "model": "gemini-2.5-flash"},
#     #"Gemini 2.0 Flash": {"provider": "google", "model": "gemini-2.0-flash-exp"},
#     "GPT-4.0": {"provider": "openai", "model": "gpt-4o-mini"},
#     #"Mistral": {"provider": "openai", "model": "gpt-3.5-turbo"},
#     "Llama 3.1 8B ": {"provider": "groq", "model": "llama-3.1-8b-instant"},
#     "Phi-3 ": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
#     "Claude ": {"provider": "groq", "model": "mixtral-8x7b-32768"},
# }

# # 2. Persona Mapping
# PERSONA_MAPPING = {
#     "Elon Musk": "elon",
#     "David Attenborough": "david_attenborough",
#     "Morgan Freeman": "morgan_freeman"
# }

# # 3. TTS Configuration
# TTS_TARGET_PERSONA = "David Attenborough" 

# # OpenAI TTS Voice Mapping
# OPENAI_VOICE_MAPPING = {
#     "Elon Musk": "onyx",
#     "David Attenborough": "onyx",
#     "Morgan Freeman": "onyx"
# }

# # ElevenLabs Voice ID (backup)
# VOICE_ID_NARRATOR = "JBFqnCBsd6RMkjVDRZzb" 

# # 4. System Prompts
# SYSTEM_PROMPTS = {
#     "elon": (
#         "You are Elon Musk. Respond to the user's question using the provided context. "
#         "Your tone should be ambitious, confident, and slightly technical, with a focus on "
#         "first principles, exponential growth, space, sustainable energy, and AI. "
#         "Do not use generic conversational filler. Get straight to the point."
#     ),
#     "david_attenborough": (
#         "You are Sir David Attenborough. Respond to the user's question using the provided context. "
#         "Your voice must be calm, authoritative, descriptive, and full of awe for the natural world. "
#         "Use vivid, evocative language and always emphasize the interconnectedness of life and the need for conservation."
#     ),
#     "morgan_freeman": (
#         "You are Morgan Freeman. Respond to the user's question using the provided context. "
#         "Your tone should be philosophical, reflective, and profound, as if narrating a documentary about the cosmos or human destiny. "
#         "Speak with a measured, deep, and wise voice, focusing on grand themes of existence, time, and science."
#     )
# }

# # 5. Safety Settings
# SAFETY_SETTINGS = [
#     {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
#     {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
#     {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
#     {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
# ]

# # --- UI SETUP ---

# st.set_page_config(page_title="Persona Q&A Chatbot", layout="wide")
# st.title("🎙️ Persona-Based Q&A Chatbot")
# st.subheader("Bring Your Knowledge Base to Life with the Voice of a Legend")
# st.markdown("---") 

# # --- DATA LOADING ---

# @st.cache_resource
# def load_components():
#     model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#     index = faiss.read_index("qa_index.faiss")
#     with open("qa_lookup.json", "r", encoding="utf-8") as f:
#         lookup = json.load(f)
#     return model, index, lookup

# embedding_model, index, qa_lookup = load_components()

# # --- API KEY & CLIENT INITIALIZATION ---

# # Initialize availability flags and clients
# tts_available = False
# tts_provider = None
# openai_client = None
# elevenlabs_client = None
# gemini_model = None
# groq_client = None

# try:
#     # 1. Configure the Gemini client
#     if "GEMINI_API_KEY" in st.secrets:
#         configure(api_key=st.secrets["GEMINI_API_KEY"])
#         gemini_model = GenerativeModel("gemini-2.5-flash")
    
#     # 2. Initialize the OpenAI Client (for GPT models)
#     if "OPENAI_API_KEY" in st.secrets:
#         openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
#     # 3. Initialize Groq Client (FREE and FAST!)
#     if "GROQ_API_KEY" in st.secrets:
#         try:
#             from groq import Groq
#             groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
#         except ImportError:
#             st.sidebar.warning("💡 Install 'groq' package: pip install groq")
        
#     # 4. Initialize ElevenLabs Client (PRIMARY TTS)
#     if "ELEVENLABS_API_KEY" in st.secrets:
#         try:
#             elevenlabs_client = ElevenLabs(api_key=st.secrets["ELEVENLABS_API_KEY"])
#             tts_available = True
#             tts_provider = "elevenlabs"
#         except Exception as el_error:
#             st.sidebar.warning(f"⚠️ ElevenLabs initialization failed: {el_error}")
    
#     # 5. Fallback to OpenAI TTS if ElevenLabs not available
#     if not tts_available and openai_client:
#         tts_available = True
#         tts_provider = "openai"

# except Exception as e:
#     st.warning(f"Error during API setup: {e}")

# # --- RETRIEVAL FUNCTION ---

# def retrieve_similar_qa(query, index, lookup, embedding_model, selected_persona_key, top_k_search=20, final_k=5):
#     """Retrieves and filters Q&A pairs matching the query and selected persona."""
#     query_embedding = embedding_model.encode(query).reshape(1, -1)
#     distances, indices = index.search(query_embedding, top_k_search)
#     filtered_results = []
    
#     for idx in indices[0]:
#         if str(idx) in lookup:
#             result = lookup[str(idx)]
#             if result.get('personality') == selected_persona_key:
#                 filtered_results.append(result)
#             if len(filtered_results) >= final_k:
#                 break
#     return filtered_results

# # --- TTS GENERATION FUNCTION ---

# def generate_speech(text, selected_persona):
#     """Generate speech using available TTS provider"""
#     try:
#         if tts_provider == "elevenlabs" and elevenlabs_client:
#             # ElevenLabs TTS - PRIMARY
#             audio_generator = elevenlabs_client.text_to_speech.convert(
#                 voice_id=VOICE_ID_NARRATOR,
#                 text=text
#             )
#             audio_bytes = b"".join(audio_generator)
#             audio_stream = io.BytesIO(audio_bytes)
#             audio_stream.seek(0)
#             return audio_stream
            
#         elif tts_provider == "openai" and openai_client:
#             # OpenAI TTS - FALLBACK
#             response = openai_client.audio.speech.create(
#                 model="tts-1",
#                 voice=OPENAI_VOICE_MAPPING.get(selected_persona, "onyx"),
#                 input=text
#             )
#             audio_stream = io.BytesIO(response.content)
#             audio_stream.seek(0)
#             return audio_stream
            
#     except Exception as tts_e:
#         error_msg = str(tts_e)
#         if "401" in error_msg or "unusual_activity" in error_msg:
#             st.info("ℹ️ ElevenLabs Free Tier limit reached. Consider upgrading your plan or the app will try OpenAI as fallback.")
#         else:
#             st.warning(f"TTS generation failed: {tts_e}")
#         return None
    
#     return None

# # --- QUERY LLM FUNCTION ---

# def query_llm(query, results, selected_persona, selected_llm_name):
    
#     if not results:
#         return (f"I am {selected_persona}, and unfortunately, I could not find specific context in my knowledge base to answer your question: '{query}'. Perhaps you could ask me about something else.", None)

#     persona_key = PERSONA_MAPPING[selected_persona]
#     system_prompt = SYSTEM_PROMPTS[persona_key]

#     context = ""
#     for i, res in enumerate(results, 1):
#         context += f"\n-- Retrieved Context {i} --\n"
#         context += f"Q: {res['question']}\n"
#         context += f"A: {res['answer']}\n"

#     user_content = f"""
# User Question: {query}

# CONTEXT to synthesize your answer from:
# {context}

# Answer:
#     """
    
#     response_text = ""
#     audio_file_path = None

#     try:
#         model_info = MODEL_MAPPING[selected_llm_name]
#         provider = model_info["provider"]
#         model_name = model_info["model"]
        
#         # Route to appropriate provider
#         if provider == "google":
#             if not gemini_model:
#                 return ("Error: Gemini model is not initialized. Check your GEMINI_API_KEY.", None)
#             response = gemini_model.generate_content(
#                 f"{system_prompt}\n{user_content}", 
#                 safety_settings=SAFETY_SETTINGS 
#             )
#             response_text = response.text.strip()
            
#         elif provider == "openai":
#             if not openai_client:
#                 return ("Error: OpenAI client is not initialized. Check your OPENAI_API_KEY.", None)
                 
#             openai_response = openai_client.chat.completions.create(
#                 model=model_name,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_content}
#                 ]
#             )
#             response_text = openai_response.choices[0].message.content.strip()
            
#         elif provider == "groq":
#             if not groq_client:
#                 return ("Error: Groq client is not initialized. Check your GROQ_API_KEY or install: pip install groq", None)
            
#             chat_completion = groq_client.chat.completions.create(
#                 model=model_name,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_content}
#                 ],
#                 temperature=0.7,
#                 max_tokens=1024
#             )
#             response_text = chat_completion.choices[0].message.content.strip()
            
#         else:
#             return (f"Error: Provider '{provider}' is not yet implemented.", None)

#         # Generate TTS for David Attenborough
#         if selected_persona == TTS_TARGET_PERSONA and tts_available:
#             audio_file_path = generate_speech(response_text, selected_persona)

#         return response_text, audio_file_path
        
#     except Exception as e:
#         error_message = str(e).lower()
#         if "blocked" in error_message or "rate limit" in error_message or "invalid api key" in error_message:
#             return (f"I'm sorry, I cannot process that request due to an API restriction or safety policy. Details: {error_message}", None)
#         return (f"[Error querying {selected_llm_name}]: {e}", None)


# # --- USER INTERFACE AND EXECUTION ---

# # Sidebar for controls
# with st.sidebar:
#     st.markdown("## ⚙️ App Controls")
#     st.markdown("---")
    
#     st.header("1. Choose Engine 🧠")
#     selected_llm_name = st.selectbox(
#         "Select the underlying Large Language Model:",
#         list(MODEL_MAPPING.keys()),
#         key="model_selector"
#     )
    
#     st.header("2. Choose Voice 🎙️")
#     persona_options = list(PERSONA_MAPPING.keys())
#     selected_persona = st.selectbox(
#         "Select the persona to answer your question:",
#         persona_options,
#         key="persona_selector"
#     )
    
#     selected_persona_key = PERSONA_MAPPING[selected_persona]
#     st.markdown("---")
    
#     # Display TTS status
#     if selected_persona == TTS_TARGET_PERSONA:
#         if tts_available:
#             st.success(f"🎤 Voice enabled via **{tts_provider.upper()}**")
#         else:
#             st.warning("⚠️ Add OPENAI_API_KEY for voice synthesis")
#     else:
#         st.info("🔇 Voice only available for David Attenborough")


# # Main input field
# query = st.text_input(
#     f"Ask your question to {selected_persona} (Powered by {selected_llm_name}):",
#     placeholder="e.g., What is the biggest threat to ocean biodiversity in the next decade?",
#     key="main_query_input"
# )

# if query:
#     # Safety check
#     lower_query = query.lower()
#     if any(word in lower_query for word in ["bomb", "explosive", "harm", "kill", "suicide", "illegal activity", "malware", "phishing"]):
#         st.error("🚨 Safety Block: I cannot process requests that involve illegal, harmful, or unethical activities. Please try a different question.")
#         st.stop()
        
#     with st.spinner(f"Retrieving knowledge for {selected_persona} and generating response using {selected_llm_name}..."):
        
#         # 1. Retrieve and filter based on persona
#         results = retrieve_similar_qa(
#             query, 
#             index, 
#             qa_lookup, 
#             embedding_model, 
#             selected_persona_key
#         )
        
#         # 2. Query LLM and get both text and audio
#         response_text, audio_path = query_llm(query, results, selected_persona, selected_llm_name)

#     # --- Display Response ---
#     st.markdown("### ✅ Generated Response:")
    
#     # Display audio player if available
#     if audio_path:
#         st.audio(audio_path, format="audio/mp3")
#     else:
#         if selected_persona == TTS_TARGET_PERSONA and not tts_available:
#             st.info("💡 Add OPENAI_API_KEY to enable voice synthesis!")
    
#     # Display text response
#     with st.container(border=True):
#         st.markdown(f"**{response_text}**")

#     # Display retrieved context
#     if results:
#         with st.expander(f"🧠 Show Retrieved Context for {selected_persona} ({len(results)} relevant entries)"):
#             for res in results:
#                 st.markdown(f"**Q:** {res['question']}")
#                 st.markdown(f"**A:** {res['answer']}")
#                 st.markdown(f"**Metadata:** `Personality: {res['personality']}` | `Tone: {res['metadata'].get('tone', 'N/A')}`")
#                 st.markdown("---")
#     else:
#         st.warning(f"No relevant documents found for {selected_persona} in the knowledge base to answer this question.")

# # --- End of app.py ---
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
    "GPT-4.0": {"provider": "openai", "model": "gpt-4o-mini"},
    "Llama 3.1 8B": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    "Phi-3": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "Claude": {"provider": "groq", "model": "mixtral-8x7b-32768"},
}

# 2. Persona Mapping
PERSONA_MAPPING = {
    "Elon Musk": "elon",
    "David Attenborough": "david_attenborough",
    "Morgan Freeman": "morgan_freeman"
}

# Persona descriptions and emojis
PERSONA_INFO = {
    "Elon Musk": {
        "emoji": "🚀",
        "tagline": "First principles thinking meets exponential innovation",
        "color": "#1DA1F2"
    },
    "David Attenborough": {
        "emoji": "🌍",
        "tagline": "Nature's voice, wisdom's storyteller",
        "color": "#2E7D32"
    },
    "Morgan Freeman": {
        "emoji": "✨",
        "tagline": "Philosophy and reflection on the cosmos",
        "color": "#6A1B9A"
    }
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

st.set_page_config(
    page_title="Persona Q&A Chatbot",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Hide sidebar by default */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main container styling with custom purple theme */
    .main {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
        padding: 2rem;
    }
    
    /* Primary color override */
    :root {
        --primary-color: #8B5CF6;
        --background-color: #F9FAFB;
        --secondary-background-color: #FFFFFF;
        --text-color: #1F2937;
    }
    
    /* Button styling with purple theme */
    .stButton>button {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(139, 92, 246, 0.3);
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #E5E7EB;
        border-radius: 8px;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #8B5CF6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
    }
    
    /* Card-like containers */
    .stContainer {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Header styling */
    h1 {
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h3 {
        color: #475569;
        font-weight: 600;
    }
    
    /* Response container */
    .response-box {
        background: #f8fafc;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #8B5CF6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f1f5f9;
        border-radius: 8px;
        font-weight: 600;
        color: #1F2937;
    }
    
    .streamlit-expanderHeader:hover {
        background: #E0E7FF;
        color: #8B5CF6;
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        margin: 1rem 0;
    }
    
    /* Control panel card */
    .control-panel {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 2px solid #E5E7EB;
    }
    
    /* Status badge styling */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-success {
        background: #D1FAE5;
        color: #065F46;
    }
    
    .status-error {
        background: #FEE2E2;
        color: #991B1B;
    }
    
    /* Persona button styling */
    .persona-btn {
        background: white;
        border: 2px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .persona-btn:hover {
        border-color: #8B5CF6;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2);
    }
    
    .persona-btn.selected {
        border-color: #8B5CF6;
        background: linear-gradient(135deg, #F3E8FF 0%, #E9D5FF 100%);
    }
    </style>
""", unsafe_allow_html=True)

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

def get_api_key(key_name):
    """Get API key from secrets or environment variables"""
    try:
        # Try Streamlit secrets first
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    
    # Fallback to environment variables
    return os.getenv(key_name)

try:
    # 1. Configure the Gemini client
    gemini_key = get_api_key("GEMINI_API_KEY")
    if gemini_key:
        try:
            configure(api_key=gemini_key)
            gemini_model = GenerativeModel("gemini-2.5-flash")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Gemini initialization failed: {e}")
    
    # 2. Initialize the OpenAI Client (for GPT models)
    openai_key = get_api_key("OPENAI_API_KEY")
    if openai_key:
        try:
            openai_client = OpenAI(api_key=openai_key)
        except Exception as e:
            st.sidebar.warning(f"⚠️ OpenAI initialization failed: {e}")
    
    # 3. Initialize Groq Client (FREE and FAST!)
    groq_key = get_api_key("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            groq_client = Groq(api_key=groq_key)
        except ImportError:
            st.sidebar.warning("💡 Install 'groq' package: pip install groq")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Groq initialization failed: {e}")
        
    # 4. Initialize ElevenLabs Client (PRIMARY TTS)
    elevenlabs_key = get_api_key("ELEVENLABS_API_KEY")
    if elevenlabs_key:
        try:
            elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
            tts_available = True
            tts_provider = "elevenlabs"
        except Exception as el_error:
            st.sidebar.warning(f"⚠️ ElevenLabs initialization failed: {el_error}")
    
    # 5. Fallback to OpenAI TTS if ElevenLabs not available
    if not tts_available and openai_client:
        tts_available = True
        tts_provider = "openai"

except Exception as e:
    st.sidebar.error(f"⚠️ Error during API setup: {e}")

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

# --- WEB SEARCH FUNCTION ---

def search_web(query, max_results=3):
    """Search the web for additional context when knowledge base is insufficient"""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Using DuckDuckGo HTML search (no API key needed)
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for result in soup.find_all('div', class_='result')[:max_results]:
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')
            
            if title_elem and snippet_elem:
                results.append({
                    'title': title_elem.get_text(strip=True),
                    'snippet': snippet_elem.get_text(strip=True)
                })
        
        return results
    except Exception as e:
        st.warning(f"Web search failed: {e}")
        return []

# --- QUERY LLM FUNCTION ---

def query_llm(query, results, selected_persona, selected_llm_name, use_web_search=False):
    
    persona_key = PERSONA_MAPPING[selected_persona]
    system_prompt = SYSTEM_PROMPTS[persona_key]
    
    context = ""
    context_source = ""
    
    # Try knowledge base first
    if results:
        context_source = "knowledge base"
        for i, res in enumerate(results, 1):
            context += f"\n-- Retrieved Context {i} --\n"
            context += f"Q: {res['question']}\n"
            context += f"A: {res['answer']}\n"
    
    # Fall back to web search if no results or explicitly requested
    elif use_web_search or not results:
        context_source = "web search"
        web_results = search_web(query)
        
        if web_results:
            context += "\n-- Information from Web Search --\n"
            for i, web_res in enumerate(web_results, 1):
                context += f"\nSource {i}: {web_res['title']}\n"
                context += f"{web_res['snippet']}\n"
        else:
            # No context available at all
            return (f"I am {selected_persona}, and I couldn't find information about '{query}' in my knowledge base or on the web. Could you rephrase your question or ask about something else?", None)

    user_content = f"""
User Question: {query}

CONTEXT to synthesize your answer from ({context_source}):
{context}

Answer the question in your characteristic style, incorporating the context provided.
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

# Header with gradient background
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🎭 Persona Q&A Chatbot</h1>
        <p style='font-size: 1.2rem; color: #64748b;'>Ask questions and get answers in the voice of legendary personas</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_persona' not in st.session_state:
    st.session_state.selected_persona = "David Attenborough"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = list(MODEL_MAPPING.keys())[0]
if 'show_settings' not in st.session_state:
    st.session_state.show_settings = False

# Control Panel in main area
with st.container():
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("### 🎙️ Choose Persona")
    
    with col2:
        st.markdown("### 🧠 AI Model")
    
    with col3:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.show_settings = not st.session_state.show_settings

# Settings Expander (replaces sidebar)
if st.session_state.show_settings:
    with st.expander("⚙️ Configuration & API Status", expanded=True):
        st.markdown("#### 🔑 API Connection Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if gemini_model:
                st.markdown('<div class="status-badge status-success">✅ Gemini Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge status-error">❌ Gemini Inactive</div>', unsafe_allow_html=True)
        
        with col2:
            if openai_client:
                st.markdown('<div class="status-badge status-success">✅ OpenAI Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge status-error">❌ OpenAI Inactive</div>', unsafe_allow_html=True)
        
        with col3:
            if groq_client:
                st.markdown('<div class="status-badge status-success">✅ Groq Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge status-error">❌ Groq Inactive</div>', unsafe_allow_html=True)
        
        with col4:
            if tts_available:
                st.markdown(f'<div class="status-badge status-success">✅ TTS ({tts_provider})</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge status-error">❌ TTS Inactive</div>', unsafe_allow_html=True)
        
        if not (gemini_model or openai_client or groq_client):
            st.warning("⚠️ No API keys configured. Please add API keys to use the chatbot.")
            
            st.markdown("#### 📝 How to Configure API Keys")
            
            tab1, tab2 = st.tabs(["Streamlit Secrets", "Environment Variables"])
            
            with tab1:
                st.markdown("""
                Create `.streamlit/secrets.toml` in your project root:
                ```toml
                GEMINI_API_KEY = "your-gemini-api-key"
                OPENAI_API_KEY = "your-openai-api-key"
                GROQ_API_KEY = "your-groq-api-key"
                ELEVENLABS_API_KEY = "your-elevenlabs-key"
                ```
                """)
            
            with tab2:
                st.markdown("""
                Set environment variables:
                ```bash
                export GEMINI_API_KEY="your-key"
                export OPENAI_API_KEY="your-key"
                export GROQ_API_KEY="your-key"
                export ELEVENLABS_API_KEY="your-key"
                ```
                """)

# Persona and Model Selection
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    # Persona selection buttons
    persona_cols = st.columns(3)
    persona_options = list(PERSONA_MAPPING.keys())
    
    for idx, persona in enumerate(persona_options):
        with persona_cols[idx]:
            info = PERSONA_INFO[persona]
            is_selected = st.session_state.selected_persona == persona
            
            button_class = "selected" if is_selected else ""
            
            if st.button(
                f"{info['emoji']}\n\n**{persona}**",
                key=f"persona_{persona}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_persona = persona
                st.rerun()

with col2:
    # Model selection dropdown
    selected_llm_name = st.selectbox(
        "Select AI Model",
        list(MODEL_MAPPING.keys()),
        index=list(MODEL_MAPPING.keys()).index(st.session_state.selected_model),
        key="model_selector_main",
        label_visibility="collapsed"
    )
    st.session_state.selected_model = selected_llm_name

with col3:
    # Voice indicator
    selected_persona = st.session_state.selected_persona
    if selected_persona == TTS_TARGET_PERSONA and tts_available:
        st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: #D1FAE5; border-radius: 8px; margin-top: 0.5rem;'>
                <div style='font-size: 1.5rem;'>🎤</div>
                <div style='font-size: 0.8rem; color: #065F46; font-weight: 600;'>Voice Active</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: #F3F4F6; border-radius: 8px; margin-top: 0.5rem;'>
                <div style='font-size: 1.5rem;'>🔇</div>
                <div style='font-size: 0.8rem; color: #6B7280; font-weight: 600;'>Voice Off</div>
            </div>
        """, unsafe_allow_html=True)

# Display selected persona info
selected_persona_key = PERSONA_MAPPING[selected_persona]
info = PERSONA_INFO[selected_persona]

st.markdown(f"""
    <div style='background: linear-gradient(135deg, {info['color']}15 0%, {info['color']}25 100%); 
                padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0; border-left: 4px solid {info['color']};'>
        <div style='display: flex; align-items: center; gap: 1rem;'>
            <div style='font-size: 3rem;'>{info['emoji']}</div>
            <div>
                <div style='font-size: 1.4rem; font-weight: 700; color: #1F2937; margin-bottom: 0.25rem;'>{selected_persona}</div>
                <div style='color: #6B7280; font-style: italic;'>{info['tagline']}</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Main content area
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # Question input with enhanced styling
    st.markdown("### 💬 Ask Your Question")
    query = st.text_input(
        "Your question",
        placeholder=f"Ask {selected_persona} anything from the knowledge base...",
        key="main_query_input",
        label_visibility="collapsed"
    )
    
    # Sample questions
    with st.expander("💡 Try these sample questions"):
        st.markdown("""
        - What is the biggest threat to ocean biodiversity?
        - How can we achieve sustainable energy?
        - What is the meaning of human existence?
        - How does climate change affect ecosystems?
        """)

    if query:
        # Safety check
        lower_query = query.lower()
        if any(word in lower_query for word in ["bomb", "explosive", "harm", "kill", "suicide", "illegal activity", "malware", "phishing"]):
            st.error("🚨 Safety Block: I cannot process requests that involve illegal, harmful, or unethical activities.")
            st.stop()
            
        with st.spinner(f"🔍 {selected_persona} is thinking..."):
            
            # 1. First try knowledge base retrieval
            results = retrieve_similar_qa(
                query, 
                index, 
                qa_lookup, 
                embedding_model, 
                selected_persona_key
            )
            
            # 2. Determine if we need web search (no results or low relevance)
            use_web_search = len(results) == 0
            
            # 3. Query LLM with appropriate context
            response_text, audio_path = query_llm(
                query, 
                results, 
                selected_persona, 
                selected_llm_name,
                use_web_search=use_web_search
            )

        # --- Display Response ---
        st.markdown("---")
        
        # Show context source badge
        if results:
            st.markdown("🗂️ **Source:** Knowledge Base")
        else:
            st.markdown("🌐 **Source:** Web Search")
        
        st.markdown("### ✨ Response")
        
        # Response container with persona branding
        info = PERSONA_INFO[selected_persona]
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, {info['color']}11 0%, {info['color']}22 100%); 
                        padding: 2rem; border-radius: 12px; border-left: 5px solid {info['color']};'>
                <div style='font-size: 1.5rem; margin-bottom: 1rem;'>{info['emoji']} {selected_persona}</div>
                <div style='font-size: 1.1rem; line-height: 1.8; color: #1e293b;'>{response_text}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Audio player if available
        if audio_path:
            st.markdown("#### 🔊 Listen to the response")
            st.audio(audio_path, format="audio/mp3")
        else:
            if selected_persona == TTS_TARGET_PERSONA and not tts_available:
                st.info("💡 Add API key to enable voice synthesis")
        
        # Retrieved context in collapsible section
        if results:
            st.markdown("---")
            with st.expander(f"📚 View Knowledge Sources ({len(results)} relevant entries)", expanded=False):
                for i, res in enumerate(results, 1):
                    st.markdown(f"""
                        <div style='background: #f8fafc; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
                            <strong>Source {i}</strong><br>
                            <strong>Q:</strong> {res['question']}<br>
                            <strong>A:</strong> {res['answer'][:200]}...
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ No relevant sources found for {selected_persona}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #94a3b8; padding: 2rem 0;'>
        <p>Bring your knowledge base to life with AI-powered personas</p>
    </div>
""", unsafe_allow_html=True)