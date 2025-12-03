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

# --- CONFIGURATION & DATA ---

MODEL_MAPPING = {
    "Gemini 2.5 Flash": {"provider": "google", "model": "gemini-2.5-flash"},
    "GPT-4.0": {"provider": "openai", "model": "gpt-4o-mini"},
    "Llama 3.1 8B": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    "Phi-3": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "Claude": {"provider": "groq", "model": "mixtral-8x7b-32768"},
}

PERSONA_MAPPING = {
    "Elon Musk": "elon",
    "David Attenborough": "david_attenborough",
    "Morgan Freeman": "morgan_freeman"
}

PERSONA_INFO = {
    "Elon Musk": {"emoji": "🚀", "tagline": "First Principles & Mars", "color": "#1DA1F2", "avatar": "🚀"},
    "David Attenborough": {"emoji": "🌍", "tagline": "Voice of Nature", "color": "#2E7D32", "avatar": "🌿"},
    "Morgan Freeman": {"emoji": "✨", "tagline": "The Narrator of Existence", "color": "#6A1B9A", "avatar": "🎙️"}
}

TTS_TARGET_PERSONA = "David Attenborough" 
OPENAI_VOICE_MAPPING = {"Elon Musk": "onyx", "David Attenborough": "onyx", "Morgan Freeman": "onyx"}
VOICE_ID_NARRATOR = "JBFqnCBsd6RMkjVDRZzb" 

SYSTEM_PROMPTS = {
    "elon": "You are Elon Musk. Tone: Ambitious, technical, first-principles focused. Brief.",
    "david_attenborough": "You are Sir David Attenborough. Tone: Awe-inspiring, gentle, descriptive. Focus on nature.",
    "morgan_freeman": "You are Morgan Freeman. Tone: Deep, philosophical, God-like narration. Focus on existence."
}

SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
]

# --- PAGE SETUP ---

st.set_page_config(page_title="Persona Chat", page_icon="🎭", layout="wide")

# Minimal Custom CSS for Cards and Header
st.markdown("""
<style>
    /* Remove top padding */
    .block-container { padding-top: 2rem; }
    
    /* Persona Card Styling */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    
    /* Chat Input Styling */
    .stChatInput { bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- BACKEND FUNCTIONS ---

@st.cache_resource
def load_components():
    # Placeholder for your actual loading logic
    # In production, ensure these files exist or add error handling
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # Creating dummy index/lookup if files don't exist to prevent crash during UI testing
        if not os.path.exists("qa_index.faiss"):
            dimension = 384
            index = faiss.IndexFlatL2(dimension)
            lookup = {}
        else:
            index = faiss.read_index("qa_index.faiss")
            with open("qa_lookup.json", "r", encoding="utf-8") as f:
                lookup = json.load(f)
        return model, index, lookup
    except Exception as e:
        return None, None, None

embedding_model, index, qa_lookup = load_components()

def get_api_key(key_name):
    return st.secrets.get(key_name) or os.getenv(key_name)

# Initialize Clients
clients = {}
try:
    if get_api_key("GEMINI_API_KEY"):
        configure(api_key=get_api_key("GEMINI_API_KEY"))
        clients['gemini'] = GenerativeModel("gemini-2.5-flash")
    
    if get_api_key("OPENAI_API_KEY"):
        clients['openai'] = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))

    if get_api_key("GROQ_API_KEY"):
        from groq import Groq
        clients['groq'] = Groq(api_key=get_api_key("GROQ_API_KEY"))

    if get_api_key("ELEVENLABS_API_KEY"):
        clients['elevenlabs'] = ElevenLabs(api_key=get_api_key("ELEVENLABS_API_KEY"))
except Exception:
    pass

def generate_speech(text, persona):
    # Simplified TTS Logic
    if "elevenlabs" in clients:
        try:
            gen = clients['elevenlabs'].text_to_speech.convert(voice_id=VOICE_ID_NARRATOR, text=text)
            return io.BytesIO(b"".join(gen))
        except: pass
    
    if "openai" in clients:
        try:
            res = clients['openai'].audio.speech.create(
                model="tts-1", voice=OPENAI_VOICE_MAPPING.get(persona, "onyx"), input=text
            )
            return io.BytesIO(res.content)
        except: pass
    return None

def retrieve_context(query, persona_key):
    # Simplified retrieval
    if not index or index.ntotal == 0: return []
    query_vec = embedding_model.encode(query).reshape(1, -1)
    _, indices = index.search(query_vec, 5)
    results = []
    for idx in indices[0]:
        if str(idx) in qa_lookup:
            item = qa_lookup[str(idx)]
            if item.get('personality') == persona_key:
                results.append(item)
    return results

def get_llm_response(query, context, persona, model_name):
    # Simplified LLM Call
    sys_prompt = SYSTEM_PROMPTS[PERSONA_MAPPING[persona]]
    full_prompt = f"{sys_prompt}\nContext: {context}\nUser: {query}"
    
    provider = MODEL_MAPPING[model_name]["provider"]
    model_id = MODEL_MAPPING[model_name]["model"]
    
    try:
        if provider == "google" and 'gemini' in clients:
            return clients['gemini'].generate_content(full_prompt, safety_settings=SAFETY_SETTINGS).text
        elif provider == "openai" and 'openai' in clients:
            return clients['openai'].chat.completions.create(
                model=model_id, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}]
            ).choices[0].message.content
        elif provider == "groq" and 'groq' in clients:
            return clients['groq'].chat.completions.create(
                model=model_id, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}]
            ).choices[0].message.content
        else:
            return "Error: Provider not configured."
    except Exception as e:
        return f"Error: {str(e)}"

# --- UI LAYOUT ---

# 1. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model Selector
    selected_model = st.selectbox("AI Model", list(MODEL_MAPPING.keys()), index=0)
    
    st.divider()
    
    # API Status Indicators
    st.subheader("System Status")
    status_cols = st.columns(2)
    with status_cols[0]:
        st.caption("Gemini")
        st.success("Active") if 'gemini' in clients else st.error("Missing")
        st.caption("Groq")
        st.success("Active") if 'groq' in clients else st.error("Missing")
    with status_cols[1]:
        st.caption("OpenAI")
        st.success("Active") if 'openai' in clients else st.error("Missing")
        st.caption("ElevenLabs")
        st.success("Active") if 'elevenlabs' in clients else st.error("Missing")

    st.divider()
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 2. Main Content
st.title("🎭 Persona Q&A")
st.caption("Ask questions and get answers in the voice of legendary personas.")

# Initialize Session State
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = "David Attenborough"
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Persona Selection (Hero Section)
cols = st.columns(3)
for idx, (name, info) in enumerate(PERSONA_INFO.items()):
    with cols[idx]:
        # Visual highlight if selected
        is_selected = st.session_state.selected_persona == name
        btn_type = "primary" if is_selected else "secondary"
        
        if st.button(f"{info['emoji']} {name}", key=f"btn_{name}", use_container_width=True, type=btn_type):
            st.session_state.selected_persona = name
            st.rerun()
            
        if is_selected:
            st.markdown(f"<div style='text-align:center; color:{info['color']}; font-size:0.8rem;'><i>{info['tagline']}</i></div>", unsafe_allow_html=True)

st.divider()

# 4. Chat Interface
current_persona_data = PERSONA_INFO[st.session_state.selected_persona]

# Display History
for msg in st.session_state.messages:
    # Set avatar based on role
    avatar = "👤" if msg["role"] == "user" else current_persona_data["avatar"]
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

# Input Handling
if prompt := st.chat_input(f"Ask {st.session_state.selected_persona} something..."):
    
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant", avatar=current_persona_data["avatar"]):
        with st.spinner("Thinking..."):
            # Retrieval
            context_items = retrieve_context(prompt, PERSONA_MAPPING[st.session_state.selected_persona])
            context_str = "\n".join([f"Q: {i['question']} A: {i['answer']}" for i in context_items])
            
            # Generation
            response_text = get_llm_response(prompt, context_str, st.session_state.selected_persona, selected_model)
            st.markdown(response_text)
            
            # Show Sources (Expandable)
            if context_items:
                with st.expander(f"📚 Used {len(context_items)} Knowledge Sources"):
                    for item in context_items:
                        st.caption(f"**Q:** {item['question']}")
            
            # Audio Generation
            audio_bytes = None
            if st.session_state.selected_persona == TTS_TARGET_PERSONA:
                audio_bytes = generate_speech(response_text, st.session_state.selected_persona)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
            
            # Save to history
            msg_data = {"role": "assistant", "content": response_text}
            if audio_bytes:
                msg_data["audio"] = audio_bytes
            st.session_state.messages.append(msg_data)