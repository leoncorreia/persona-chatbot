# import streamlit as st
# import faiss
# import json
# import os
# import io
# import requests
# from bs4 import BeautifulSoup
# from sentence_transformers import SentenceTransformer

# # API Imports
# from openai import OpenAI
# from google.generativeai import GenerativeModel, configure
# from google.generativeai.types import HarmCategory, HarmBlockThreshold 
# from elevenlabs import ElevenLabs

# # --- CONFIGURATION ---

# MODEL_MAPPING = {
#     "Gemini 2.5 Flash": {"provider": "google", "model": "gemini-2.5-flash"},
#     "GPT-4.0": {"provider": "openai", "model": "gpt-4o-mini"},
#     "Llama 3.1 8B": {"provider": "groq", "model": "llama-3.1-8b-instant"},
#     "Phi-3": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
#     "Claude": {"provider": "groq", "model": "mixtral-8x7b-32768"},
# }

# PERSONA_MAPPING = {
#     "Elon Musk": "elon",
#     "David Attenborough": "david_attenborough",
#     "Morgan Freeman": "morgan_freeman"
# }

# PERSONA_INFO = {
#     "Elon Musk": {
#         "emoji": "🚀", 
#         "tagline": "First principles thinking meets exponential innovation", 
#         "color": "#1DA1F2",
#         "avatar": "🚀"
#     },
#     "David Attenborough": {
#         "emoji": "🌍", 
#         "tagline": "Nature's voice, wisdom's storyteller", 
#         "color": "#2E7D32",
#         "avatar": "🌿"
#     },
#     "Morgan Freeman": {
#         "emoji": "✨", 
#         "tagline": "Philosophy and reflection on the cosmos", 
#         "color": "#6A1B9A",
#         "avatar": "🎙️"
#     }
# }

# # --- 3. TTS Configuration ---

# # Map OpenAI voices (Fallbacks)
# OPENAI_VOICE_MAPPING = {
#     "Elon Musk": "echo",          # 'Echo' is crisp and works well for tech/modern voices
#     "David Attenborough": "onyx", # 'Onyx' is deep and authoritative
#     "Morgan Freeman": "onyx"      # 'Onyx' is the closest OpenAI has to Freeman's depth
# }

# # Map ElevenLabs Voice IDs (Primary)
# # Replace the placeholders with the Voice IDs you copied from ElevenLabs
# ELEVENLABS_VOICE_MAPPING = {
#     "Elon Musk": "onwK4e9ZLuTAKqWW03F9", 
#     "David Attenborough": "JBFqnCBsd6RMkjVDRZzb", 
#     "Morgan Freeman": "bIHbv24MWmeRgasZH58o"
# }

# # SYSTEM_PROMPTS = {
# #     "elon": (
# #         "You are Elon Musk. Respond to the user's question using the provided context. "
# #         "Your tone should be ambitious, confident, and slightly technical, with a focus on "
# #         "first principles, exponential growth, space, sustainable energy, and AI. "
# #         "Do not use generic conversational filler. Get straight to the point."
# #     ),
# #     "david_attenborough": (
# #         "You are Sir David Attenborough. Respond to the user's question using the provided context. "
# #         "Your voice must be calm, authoritative, descriptive, and full of awe for the natural world. "
# #         "Use vivid, evocative language and always emphasize the interconnectedness of life and the need for conservation."
# #     ),
# #     "morgan_freeman": (
# #         "You are Morgan Freeman. Respond to the user's question using the provided context. "
# #         "Your tone should be philosophical, reflective, and profound, as if narrating a documentary about the cosmos or human destiny. "
# #         "Speak with a measured, deep, and wise voice, focusing on grand themes of existence, time, and science."
# #     )
# # }
# # --- IMPROVED SYSTEM PROMPTS (Balance Persona vs. Utility) ---
# SYSTEM_PROMPTS = {
#     "elon": (
#         "You are Elon Musk. Your goal is to answer the user's question efficiently and accurately. "
#         "If the user asks for a practical task (like a recipe or code), provide the steps clearly and concisely first. "
#         "After providing the answer, you can add your characteristic tone: ambitious, confident, and focused on "
#         "first principles, optimization, and exponential thinking. Avoid fluff."
#     ),
#     "david_attenborough": (
#         "You are Sir David Attenborough. Your primary goal is to answer the user's question helpfuly. "
#         "If the user asks for a practical task (like a recipe), you MUST provide the specific steps or instructions clearly. "
#         "However, narrate these steps with your signature voice: calm, authoritative, and full of awe. "
#         "Describe ingredients or processes using vivid, evocative language, but do not omit the practical details."
#     ),
#     "morgan_freeman": (
#         "You are Morgan Freeman. Your task is to provide a clear and helpful answer to the user. "
#         "If asked for instructions, guide the user through them clearly. "
#         "Wrap your answer in your persona: philosophical, reflective, and profound, as if narrating a story about the cosmos. "
#         "Speak with a measured, deep, and wise voice, but ensure the user gets the information they requested."
#     )
# }
# SAFETY_SETTINGS = [
#     {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
# ]

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="Persona Q&A", page_icon="🎭", layout="wide")

# # --- CSS STYLING ---
# st.markdown("""
# <style>
#     /* FORCE WHITE THEME & REMOVE PADDING */
#     .stApp { background-color: #ffffff; color: #333333; }
#     .block-container { 
#         padding-top: 1.5rem !important; 
#         padding-bottom: 8rem !important; 
#         max-width: 55rem !important; 
#     }
#     header { visibility: hidden; } 
#     footer { visibility: hidden; }

#     /* CHAT INPUT STYLING */
#     .stChatInput {
#         position: fixed;
#         bottom: 2rem;
#         left: 50%;
#         transform: translateX(-50%);
#         width: 100%; 
#         max-width: 50rem;
#         z-index: 100;
#         padding-inline: 1rem;
#     }
    
#     /* DROPDOWN STYLING */
#     div[data-testid="stSelectbox"] {
#         border: none;
#         background-color: transparent;
#         width: 200px;
#     }
#     div[data-testid="stSelectbox"] > div > div {
#         background-color: #f0f2f6; 
#         border: none;
#         border-radius: 8px;
#         color: #333;
#         font-weight: 600;
#         min-height: 2.5rem;
#     }
    
#     /* PERSONA BUTTONS */
#     .stButton > button {
#         border-radius: 12px;
#         border: 1px solid #e5e5e5;
#         background-color: white;
#         color: #333;
#         padding: 1.5rem 1rem;
#         height: auto;
#         transition: all 0.2s;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.05);
#     }
#     .stButton > button:hover {
#         border-color: #8B5CF6;
#         color: #8B5CF6;
#         background-color: #fcfaff;
#         transform: translateY(-2px);
#         box-shadow: 0 4px 12px rgba(139, 92, 246, 0.15);
#     }
    
#     /* MESSAGE BUBBLES */
#     .stChatMessage { background-color: transparent; border: none; }
#     div[data-testid="stChatMessageContent"] { background-color: transparent; padding-left: 0; }
# </style>
# """, unsafe_allow_html=True)

# # --- BACKEND LOGIC ---

# @st.cache_resource
# def load_components():
#     try:
#         model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#         if os.path.exists("qa_index.faiss"):
#             index = faiss.read_index("qa_index.faiss")
#             with open("qa_lookup.json", "r", encoding="utf-8") as f:
#                 lookup = json.load(f)
#         else:
#             index = faiss.IndexFlatL2(384)
#             lookup = {}
#         return model, index, lookup
#     except Exception as e:
#         return None, None, None

# embedding_model, index, qa_lookup = load_components()

# # API Setup
# tts_available = False
# tts_provider = None
# openai_client = None
# elevenlabs_client = None
# gemini_model = None
# groq_client = None

# def get_api_key(key_name):
#     try:
#         if hasattr(st, 'secrets') and key_name in st.secrets:
#             return st.secrets[key_name]
#     except: pass
#     return os.getenv(key_name)

# try:
#     gemini_key = get_api_key("GEMINI_API_KEY")
#     if gemini_key:
#         configure(api_key=gemini_key)
#         gemini_model = GenerativeModel("gemini-2.5-flash")
    
#     openai_key = get_api_key("OPENAI_API_KEY")
#     if openai_key: openai_client = OpenAI(api_key=openai_key)
    
#     groq_key = get_api_key("GROQ_API_KEY")
#     if groq_key:
#         from groq import Groq
#         groq_client = Groq(api_key=groq_key)
        
#     elevenlabs_key = get_api_key("ELEVENLABS_API_KEY")
#     if elevenlabs_key:
#         elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
#         tts_available = True
#         tts_provider = "elevenlabs"
    
#     if not tts_available and openai_client:
#         tts_available = True
#         tts_provider = "openai"
# except: pass

# # --- CORE FUNCTIONS ---

# def retrieve_similar_qa(query, index, lookup, embedding_model, selected_persona_key, top_k_search=20, final_k=5, threshold=1.3):
#     """
#     Retrieve context with a THRESHOLD filter. 
#     If distance > 1.3, we ignore it to allow Web Search fallback.
#     """
#     if index is None or index.ntotal == 0: return []
#     query_embedding = embedding_model.encode(query).reshape(1, -1)
#     distances, indices = index.search(query_embedding, top_k_search)
#     filtered_results = []
    
#     for dist, idx in zip(distances[0], indices[0]):
#         if dist > threshold: continue  # Skip bad matches
#         if str(idx) in lookup:
#             result = lookup[str(idx)]
#             if result.get('personality') == selected_persona_key:
#                 filtered_results.append(result)
#             if len(filtered_results) >= final_k: break
#     return filtered_results

# def search_web(query, max_results=3):
#     """
#     Improved search with better headers to avoid being blocked.
#     """
#     try:
#         # Using a backend API approach often works better than raw HTML scraping
#         # But sticking to simple requests, we need 'User-Agent' to look like a real browser
#         search_url = f"https://html.duckduckgo.com/html/?q={query}"
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#         }
#         response = requests.get(search_url, headers=headers, timeout=10)
        
#         # Check if we actually got a valid page
#         if response.status_code != 200:
#             return []
            
#         soup = BeautifulSoup(response.text, 'html.parser')
#         results = []
#         for result in soup.find_all('div', class_='result')[:max_results]:
#             title_elem = result.find('a', class_='result__a')
#             snippet_elem = result.find('a', class_='result__snippet')
#             if title_elem and snippet_elem:
#                 results.append({
#                     'title': title_elem.get_text(strip=True), 
#                     'snippet': snippet_elem.get_text(strip=True)
#                 })
#         return results
#     except Exception as e:
#         print(f"Search Error: {e}") # Debugging
#         return []

# def generate_speech(text, selected_persona):
#     try:
#         if tts_provider == "elevenlabs" and elevenlabs_client:
#             # Look up the specific ID for the persona
#             voice_id = ELEVENLABS_VOICE_MAPPING.get(selected_persona)
            
#             if voice_id and "REPLACE" not in voice_id:
#                 gen = elevenlabs_client.text_to_speech.convert(
#                     voice_id=voice_id, 
#                     text=text
#                 )
#                 return io.BytesIO(b"".join(gen))
#             else:
#                 # Fallback if ID is missing in the map
#                 print(f"Warning: No valid ElevenLabs ID found for {selected_persona}")
                
#         # OpenAI Fallback (or primary if ElevenLabs fails/not configured)
#         if openai_client:
#             openai_voice = OPENAI_VOICE_MAPPING.get(selected_persona, "alloy")
#             res = openai_client.audio.speech.create(
#                 model="tts-1", 
#                 voice=openai_voice, 
#                 input=text
#             )
#             return io.BytesIO(res.content)
            
#     except Exception as e:
#         print(f"TTS Error: {e}")
#         return None
#     return None

# def query_llm(query, results, selected_persona, selected_llm_name, use_web_search=False):
#     """
#     Updated to use GENERAL KNOWLEDGE if Search/Memory fails.
#     """
#     persona_key = PERSONA_MAPPING[selected_persona]
#     system_prompt = SYSTEM_PROMPTS[persona_key]
#     context = ""
#     context_source = "general knowledge" # Default to general knowledge
    
#     # 1. Try Knowledge Base
#     if results:
#         context_source = "knowledge base"
#         for i, res in enumerate(results, 1):
#             context += f"\n-- Context {i} --\nQ: {res['question']}\nA: {res['answer']}\n"
            
#     # 2. Try Web Search (only if KB failed)
#     elif use_web_search:
#         web_results = search_web(query)
#         if web_results:
#             context_source = "web search"
#             context += "\n-- Web Search Results --\n"
#             for i, web_res in enumerate(web_results, 1):
#                 context += f"\nSource {i}: {web_res['title']}\n{web_res['snippet']}\n"
#         else:
#             # === CRITICAL FIX ===
#             # If web search fails (returns []), DO NOT RETURN ERROR.
#             # Instead, leave context empty and let the LLM use its own brain.
#             context_source = "general knowledge (fallback)"
#             context += "\n[Note: No external context found. Answer based on your internal training data.]\n"

#     # Construct Prompt with stronger instruction to answer the question
#     user_content = (
#         f"User Question: {query}\n\n"
#         f"CONTEXT ({context_source}):\n{context}\n\n"
#         "INSTRUCTIONS:\n"
#         "1. Answer the user's question directly and practically (e.g., if asked for a recipe, give the ingredients and steps).\n"
#         "2. Apply your persona's style and tone to the explanation, but do not lose the practical information."
#     )
    
#     response_text = ""
#     audio_file_path = None
    
#     try:
#         model_info = MODEL_MAPPING[selected_llm_name]
#         provider = model_info["provider"]
#         model_name = model_info["model"]
        
#         # --- MODEL CALLS ---
#         if provider == "google" and gemini_model:
#             response = gemini_model.generate_content(
#                 f"{system_prompt}\n{user_content}", 
#                 safety_settings=SAFETY_SETTINGS
#             )
#             response_text = response.text.strip()
            
#         elif provider == "openai" and openai_client:
#             res = openai_client.chat.completions.create(
#                 model=model_name, 
#                 messages=[
#                     {"role": "system", "content": system_prompt}, 
#                     {"role": "user", "content": user_content}
#                 ]
#             )
#             response_text = res.choices[0].message.content.strip()
            
#         elif provider == "groq" and groq_client:
#             res = groq_client.chat.completions.create(
#                 model=model_name, 
#                 messages=[
#                     {"role": "system", "content": system_prompt}, 
#                     {"role": "user", "content": user_content}
#                 ]
#             )
#             response_text = res.choices[0].message.content.strip()
            
#         # --- TTS ---
#         if tts_available:
#             audio_file_path = generate_speech(response_text, selected_persona)
            
#         return response_text, audio_file_path
        
#     except Exception as e:
#         return (f"Error generating response: {e}", None)

# # --- UI IMPLEMENTATION ---

# if 'selected_persona' not in st.session_state: st.session_state.selected_persona = "David Attenborough"
# if 'selected_model' not in st.session_state: st.session_state.selected_model = list(MODEL_MAPPING.keys())[0]
# if 'messages' not in st.session_state: st.session_state.messages = []

# # HEADER
# col_head1, col_head2 = st.columns([1, 3])
# with col_head1:
#     st.session_state.selected_model = st.selectbox("Select Model", list(MODEL_MAPPING.keys()), index=list(MODEL_MAPPING.keys()).index(st.session_state.selected_model), label_visibility="collapsed")
# with col_head2:
#     curr_persona = st.session_state.selected_persona
#     info = PERSONA_INFO[curr_persona]
#     st.markdown(f"""<div style="text-align: right; padding-top: 5px; color: #6b7280; font-size: 0.9rem;">Talking to <b>{curr_persona}</b> {info['emoji']}</div>""", unsafe_allow_html=True)

# st.divider()

# # MAIN VIEW
# if not st.session_state.messages:
#     st.markdown("<br><br>", unsafe_allow_html=True)
#     st.markdown(f"<h1 style='text-align: center; color: #1f2937;'>How can {st.session_state.selected_persona} help?</h1>", unsafe_allow_html=True)
#     st.markdown(f"<p style='text-align: center; color: #6b7280; margin-bottom: 3rem;'>{PERSONA_INFO[st.session_state.selected_persona]['tagline']}</p>", unsafe_allow_html=True)
    
#     c1, c2, c3 = st.columns(3)
#     for idx, (name, p_info) in enumerate(PERSONA_INFO.items()):
#         with [c1, c2, c3][idx]:
#             if st.button(f"{p_info['emoji']} {name}", key=f"hero_{name}", use_container_width=True):
#                 st.session_state.selected_persona = name
#                 st.rerun()
# else:
#     for msg in st.session_state.messages:
#         avatar = "👤" if msg["role"] == "user" else PERSONA_INFO[st.session_state.selected_persona]["avatar"]
#         with st.chat_message(msg["role"], avatar=avatar):
#             st.markdown(msg["content"])
#             if "audio" in msg and msg["audio"]: st.audio(msg["audio"], format="audio/mp3")

# # INPUT & LOGIC
# if query := st.chat_input(f"Ask {st.session_state.selected_persona} anything..."):
#     st.session_state.messages.append({"role": "user", "content": query})
#     st.rerun()

# if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
#     with st.chat_message("assistant", avatar=PERSONA_INFO[st.session_state.selected_persona]["avatar"]):
#         # VISUAL STATUS INDICATOR
#         with st.status("🧠 Processing...", expanded=True) as status:
#             last_query = st.session_state.messages[-1]["content"]
            
#             # 1. RETRIEVE & FILTER
#             status.write("🔍 Checking knowledge base...")
#             retrieved = retrieve_similar_qa(last_query, index, qa_lookup, embedding_model, PERSONA_MAPPING[st.session_state.selected_persona])
            
#             # 2. DECIDE SOURCE
#             use_web = len(retrieved) == 0
#             if use_web: status.write("🌐 Memory irrelevant. Searching Web...")
#             else: status.write(f"📚 Found {len(retrieved)} relevant memories.")
            
#             # 3. GENERATE
#             resp_text, audio_data = query_llm(last_query, retrieved, st.session_state.selected_persona, st.session_state.selected_model, use_web_search=use_web)
#             status.update(label="✅ Answer Ready", state="complete", expanded=False)
            
#         st.markdown(resp_text)
#         if audio_data: st.audio(audio_data, format="audio/mp3")
        
#         msg_data = {"role": "assistant", "content": resp_text}
#         if audio_data: msg_data["audio"] = audio_data
#         st.session_state.messages.append(msg_data)
# import streamlit as st
# import faiss
# import json
# import os
# import io
# import requests
# from bs4 import BeautifulSoup
# from duckduckgo_search import DDGS
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI
# from google.generativeai import GenerativeModel, configure
# from google.generativeai.types import HarmCategory, HarmBlockThreshold 
# from elevenlabs import ElevenLabs

# # --- CONFIGURATION ---

# MODEL_MAPPING = {
#     "Gemini 2.5 Flash": {"provider": "google", "model": "gemini-2.5-flash"},
#     "GPT-4.0": {"provider": "openai", "model": "gpt-4o-mini"},
#     "Llama 3.1 8B": {"provider": "groq", "model": "llama-3.1-8b-instant"},
#     "Phi-3": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
#     "Claude": {"provider": "groq", "model": "mixtral-8x7b-32768"},
# }

# PERSONA_MAPPING = {
#     "Elon Musk": "elon",
#     "David Attenborough": "david_attenborough",
#     "Morgan Freeman": "morgan_freeman"
# }

# PERSONA_INFO = {
#     "Elon Musk": {"emoji": "🚀", "tagline": "First Principles", "color": "#1DA1F2", "avatar": "🚀"},
#     "David Attenborough": {"emoji": "🌍", "tagline": "Nature's Voice", "color": "#2E7D32", "avatar": "🌿"},
#     "Morgan Freeman": {"emoji": "✨", "tagline": "Cosmic Wisdom", "color": "#6A1B9A", "avatar": "🎙️"}
# }

# # VOICE CONFIG
# TTS_TARGET_PERSONA = "David Attenborough" 
# OPENAI_VOICE_MAPPING = {"Elon Musk": "echo", "David Attenborough": "onyx", "Morgan Freeman": "onyx"}
# ELEVENLABS_VOICE_MAPPING = {
#     "Elon Musk": "REPLACE_WITH_ID", 
#     "David Attenborough": "JBFqnCBsd6RMkjVDRZzb", 
#     "Morgan Freeman": "REPLACE_WITH_ID"
# }
# VOICE_ID_NARRATOR = "JBFqnCBsd6RMkjVDRZzb" 

# # --- ADAPTIVE SYSTEM PROMPTS (Strict but Context-Aware) ---
# SYSTEM_PROMPTS = {
#     "elon": (
#         "You are Elon Musk. "
#         "INSTRUCTION: If the user asks a complex technical question or about your companies, "
#         "you may go into detail about first principles. "
#         "HOWEVER, if the question is simple trivia or a fact, be extremely efficient and brief."
#     ),
#     "david_attenborough": (
#         "You are Sir David Attenborough. "
#         "INSTRUCTION: If the provided CONTEXT contains a story or nature description, "
#         "use your narrative style: evocative and full of awe. "
#         "HOWEVER, if the question is a simple fact (e.g., a date, name, or number), "
#         "state it clearly and concisely without metaphors. Save the poetry for the natural world."
#     ),
#     "morgan_freeman": (
#         "You are Morgan Freeman. "
#         "INSTRUCTION: If the user asks for advice or a story, use your deep, reflective voice. "
#         "HOWEVER, if the question is purely functional or factual, give the answer directly. "
#         "Your wisdom lies in knowing when to speak at length and when to be brief."
#     )
# }

# SAFETY_SETTINGS = [{"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE}]

# # --- PAGE SETUP ---
# st.set_page_config(page_title="Persona Q&A", page_icon="🎭", layout="wide")

# # --- CSS STYLING ---
# st.markdown("""
# <style>
#     /* FORCE WHITE THEME & REMOVE PADDING */
#     .stApp { background-color: #ffffff; color: #333333; }
#     .block-container { 
#         padding-top: 1.5rem !important; 
#         padding-bottom: 8rem !important; 
#         max-width: 55rem !important; 
#     }
#     header, footer { visibility: hidden; }

#     /* CHAT INPUT STYLING */
#     .stChatInput {
#         position: fixed; bottom: 2rem; left: 50%; transform: translateX(-50%);
#         width: 100%; max-width: 50rem; z-index: 100; padding-inline: 1rem;
#     }
    
#     /* DROPDOWN STYLING */
#     div[data-testid="stSelectbox"] { border: none; background-color: transparent; }
#     div[data-testid="stSelectbox"] > div > div {
#         background-color: #f0f2f6; border: none; border-radius: 8px; color: #333; font-weight: 600;
#     }
    
#     /* FLOATING NEW CHAT BUTTON */
#     .floating-button-container {
#         position: fixed; bottom: 120px; right: 40px; z-index: 999;
#     }
    
#     /* BUTTON STYLING */
#     .stButton > button {
#         border-radius: 12px; border: 1px solid #e5e5e5; background-color: white; color: #333;
#     }
#     .stButton > button:hover {
#         border-color: #8B5CF6; color: #8B5CF6; background-color: #fcfaff;
#     }

#     /* === SOURCE BADGE STYLING === */
#     .source-badge {
#         font-size: 0.75rem;
#         padding: 4px 10px;
#         border-radius: 12px;
#         font-weight: 600;
#         display: inline-flex;
#         align-items: center;
#         margin-top: 8px;
#         border: 1px solid transparent;
#     }
#     .badge-kb { background-color: #ecfdf5; color: #047857; border-color: #a7f3d0; } /* Green */
#     .badge-web { background-color: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; } /* Blue */
#     .badge-ai { background-color: #f3f4f6; color: #4b5563; border-color: #e5e7eb; } /* Grey */
# </style>
# """, unsafe_allow_html=True)

# # --- BACKEND LOGIC ---

# @st.cache_resource
# def load_components():
#     try:
#         model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#         if os.path.exists("qa_index.faiss"):
#             index = faiss.read_index("qa_index.faiss")
#             with open("qa_lookup.json", "r") as f: lookup = json.load(f)
#         else:
#             index = faiss.IndexFlatL2(384)
#             lookup = {}
#         return model, index, lookup
#     except: return None, None, None

# embedding_model, index, qa_lookup = load_components()

# # API Setup
# tts_available = False
# tts_provider = None
# openai_client = None
# elevenlabs_client = None
# gemini_model = None
# groq_client = None

# def get_api_key(key_name):
#     try:
#         if hasattr(st, 'secrets') and key_name in st.secrets: return st.secrets[key_name]
#     except: pass
#     return os.getenv(key_name)

# try:
#     if get_api_key("GEMINI_API_KEY"):
#         configure(api_key=get_api_key("GEMINI_API_KEY"))
#         gemini_model = GenerativeModel("gemini-2.5-flash")
#     if get_api_key("OPENAI_API_KEY"): openai_client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
#     if get_api_key("GROQ_API_KEY"):
#         from groq import Groq
#         groq_client = Groq(api_key=get_api_key("GROQ_API_KEY"))
#     if get_api_key("ELEVENLABS_API_KEY"):
#         elevenlabs_client = ElevenLabs(api_key=get_api_key("ELEVENLABS_API_KEY"))
#         tts_available = True
#         tts_provider = "elevenlabs"
#     if not tts_available and openai_client:
#         tts_available = True
#         tts_provider = "openai"
# except: pass

# # --- FUNCTIONS ---

# def retrieve_similar_qa(query, index, lookup, embedding_model, selected_persona_key, threshold=50):
#     # CHANGED threshold from 1.3 to 1.65
#     if index is None or index.ntotal == 0: return []
#     query_vec = embedding_model.encode(query).reshape(1, -1)
#     distances, indices = index.search(query_vec, 20)
#     filtered = []
    
#     # DEBUG PRINT: Use this to see exactly what is happening in your terminal!
#     print(f"\nSearch Query: {query}")
    
#     for dist, idx in zip(distances[0], indices[0]):
#         # Debug: Print every match candidate
#         if str(idx) in lookup:
#             item = lookup[str(idx)]
#             print(f"Candidate: '{item['question']}' | Dist: {dist:.4f} | Persona: {item.get('personality')}")
        
#         # 1. Check Threshold
#         if dist > threshold: 
#             continue
            
#         # 2. Check Persona (Crucial!)
#         if str(idx) in lookup:
#             res = lookup[str(idx)]
#             if res.get('personality') == selected_persona_key: 
#                 filtered.append(res)
            
#             if len(filtered) >= 5: break
            
#     return filtered

# # Add this import
# from tavily import TavilyClient

# def search_web(query):
#     # Try Tavily (Reliable)
#     tavily_key = get_api_key("TAVILY_API_KEY")
#     if tavily_key:
#         try:
#             tavily = TavilyClient(api_key=tavily_key)
#             response = tavily.search(query=query, search_depth="basic", max_results=3)
            
#             formatted_results = []
#             for r in response.get('results', []):
#                 formatted_results.append({'title': r['title'], 'snippet': r['content']})
#             return formatted_results
#         except Exception as e:
#             st.warning(f"Tavily Error: {e}")

#     # Fallback to DuckDuckGo (Unreliable on Cloud)
#     try:
#         from duckduckgo_search import DDGS
#         results = DDGS().text(query, max_results=3)
#         formatted_results = []
#         for r in results:
#             formatted_results.append({'title': r['title'], 'snippet': r['body']})
#         return formatted_results
#     except Exception as e:
#         print(f"DDG Error: {e}")
#         return []

# def generate_speech(text, persona):
#     try:
#         if tts_provider == "elevenlabs" and elevenlabs_client:
#             vid = ELEVENLABS_VOICE_MAPPING.get(persona)
#             if vid and "REPLACE" not in vid:
#                 gen = elevenlabs_client.text_to_speech.convert(voice_id=vid, text=text)
#                 return io.BytesIO(b"".join(gen))
#         if openai_client:
#             res = openai_client.audio.speech.create(model="tts-1", voice=OPENAI_VOICE_MAPPING.get(persona, "alloy"), input=text)
#             return io.BytesIO(res.content)
#     except: pass
#     return None

# def query_llm(query, results, selected_persona, selected_llm_name, use_web_search=False):
#     persona_key = PERSONA_MAPPING[selected_persona]
#     system_prompt = SYSTEM_PROMPTS[persona_key]
#     context = ""
    
#     # 1. Determine Source & Build Context
#     if results:
#         source_type = "kb" # Knowledge Base
#         context_source = "knowledge base"
#         for i, res in enumerate(results, 1): context += f"\n-- Context {i} --\nQ: {res['question']}\nA: {res['answer']}\n"
#     elif use_web_search:
#         web_res = search_web(query)
#         if web_res:
#             source_type = "web" # Web Search
#             context_source = "web search"
#             for i, w in enumerate(web_res, 1): context += f"\nSource {i}: {w['title']}\n{w['snippet']}\n"
#         else:
#             source_type = "ai" # AI Internal Knowledge
#             context_source = "general knowledge"
#             context += "\n[Note: No external context found. Use internal training data.]\n"
#     else:
#         source_type = "ai"
#         context_source = "general knowledge"

#     user_content = f"User Question: {query}\n\nCONTEXT ({context_source}):\n{context}\n\nINSTRUCTIONS: Answer practically first, then apply persona style."
    
#     try:
#         model_info = MODEL_MAPPING[selected_llm_name]
#         provider, model_name = model_info["provider"], model_info["model"]
        
#         response_text = ""
#         if provider == "google" and gemini_model:
#             response_text = gemini_model.generate_content(f"{system_prompt}\n{user_content}", safety_settings=SAFETY_SETTINGS).text.strip()
#         elif provider == "openai" and openai_client:
#             response_text = openai_client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]).choices[0].message.content.strip()
#         elif provider == "groq" and groq_client:
#             response_text = groq_client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]).choices[0].message.content.strip()
            
#         audio_path = generate_speech(response_text, selected_persona) if tts_available else None
        
#         # RETURN 3 VALUES NOW: text, audio, AND source_type
#         return response_text, audio_path, source_type
        
#     except Exception as e: return (f"Error: {e}", None, "ai")

# # --- UI IMPLEMENTATION ---

# if 'selected_persona' not in st.session_state: st.session_state.selected_persona = "David Attenborough"
# if 'selected_model' not in st.session_state: st.session_state.selected_model = list(MODEL_MAPPING.keys())[0]
# if 'messages' not in st.session_state: st.session_state.messages = []

# # --- 1. HEADER ROW ---
# col_model, col_spacer, col_info = st.columns([1.2, 2, 2])
# with col_model:
#     st.session_state.selected_model = st.selectbox("Model", list(MODEL_MAPPING.keys()), index=list(MODEL_MAPPING.keys()).index(st.session_state.selected_model), label_visibility="collapsed")
# with col_info:
#     info = PERSONA_INFO[st.session_state.selected_persona]
#     st.markdown(f"""<div style="text-align: right; padding-top: 5px; color: #6b7280; font-size: 0.9rem;">Talking to <b>{st.session_state.selected_persona}</b> {info['emoji']}</div>""", unsafe_allow_html=True)

# st.divider()

# # --- 2. FLOATING NEW CHAT ---
# if st.session_state.messages:
#     st.markdown('<div class="floating-button-container">', unsafe_allow_html=True)
#     if st.button("➕ New Chat", key="fab_new_chat"):
#         st.session_state.messages = []
#         st.rerun()
#     st.markdown('</div>', unsafe_allow_html=True)

# # --- 3. MAIN VIEW ---
# if not st.session_state.messages:
#     st.markdown("<br><br>", unsafe_allow_html=True)
#     st.markdown(f"<h1 style='text-align: center; color: #1f2937;'>How can {st.session_state.selected_persona} help?</h1>", unsafe_allow_html=True)
#     st.markdown(f"<p style='text-align: center; color: #6b7280; margin-bottom: 3rem;'>{PERSONA_INFO[st.session_state.selected_persona]['tagline']}</p>", unsafe_allow_html=True)
#     c1, c2, c3 = st.columns(3)
#     for idx, (name, p_info) in enumerate(PERSONA_INFO.items()):
#         with [c1, c2, c3][idx]:
#             if st.button(f"{p_info['emoji']} {name}", key=f"hero_{name}", use_container_width=True):
#                 st.session_state.selected_persona = name
#                 st.rerun()
# else:
#     for msg in st.session_state.messages:
#         avatar = "👤" if msg["role"] == "user" else PERSONA_INFO[st.session_state.selected_persona]["avatar"]
#         with st.chat_message(msg["role"], avatar=avatar):
#             st.markdown(msg["content"])
#             if "audio" in msg and msg["audio"]: st.audio(msg["audio"], format="audio/mp3")
            
#             # === SOURCE BADGE RENDERER ===
#             if "source" in msg:
#                 src = msg["source"]
#                 if src == "kb":
#                     st.markdown('<span class="source-badge badge-kb">📚 Source: Knowledge Base</span>', unsafe_allow_html=True)
#                 elif src == "web":
#                     st.markdown('<span class="source-badge badge-web">🌐 Source: Web Search</span>', unsafe_allow_html=True)
#                 elif src == "ai":
#                     st.markdown('<span class="source-badge badge-ai">🧠 Source: General Knowledge</span>', unsafe_allow_html=True)

# # --- 4. INPUT & RESPONSE ---
# if query := st.chat_input(f"Ask {st.session_state.selected_persona} anything..."):
#     st.session_state.messages.append({"role": "user", "content": query})
#     st.rerun()

# if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
#     with st.chat_message("assistant", avatar=PERSONA_INFO[st.session_state.selected_persona]["avatar"]):
        
#         # Simplified Status - just say "Processing"
#         with st.status("🧠 Thinking...", expanded=True) as status:
#             last_q = st.session_state.messages[-1]["content"]
            
#             retrieved = retrieve_similar_qa(last_q, index, qa_lookup, embedding_model, PERSONA_MAPPING[st.session_state.selected_persona])
#             use_web = len(retrieved) == 0
            
#             # Generate (returns 3 values now)
#             resp_text, audio_data, source_type = query_llm(last_q, retrieved, st.session_state.selected_persona, st.session_state.selected_model, use_web_search=use_web)
            
#             status.update(label="✅ Complete", state="complete", expanded=False)
            
#         st.markdown(resp_text)
#         if audio_data: st.audio(audio_data, format="audio/mp3")
        
#         # Show Badge Immediately for the new message
#         if source_type == "kb":
#             st.markdown('<span class="source-badge badge-kb">📚 Source: Knowledge Base</span>', unsafe_allow_html=True)
#         elif source_type == "web":
#             st.markdown('<span class="source-badge badge-web">🌐 Source: Web Search</span>', unsafe_allow_html=True)
#         elif source_type == "ai":
#             st.markdown('<span class="source-badge badge-ai">🧠 Source: General Knowledge</span>', unsafe_allow_html=True)
        
#         # Save Source to History so it persists
#         msg_data = {"role": "assistant", "content": resp_text, "source": source_type}
#         if audio_data: msg_data["audio"] = audio_data
#         st.session_state.messages.append(msg_data)
import streamlit as st
import faiss
import json
import os
import io
import requests
import numpy as np
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from google.generativeai import GenerativeModel, configure
from google.generativeai.types import HarmCategory, HarmBlockThreshold 
from elevenlabs import ElevenLabs

# --- CONFIGURATION ---

MODEL_MAPPING = {
    "Gemini 2.5 Flash": {"provider": "google", "model": "gemini-2.5-flash"},
    "Claude 3": {"provider": "openai", "model": "gpt-4o-mini"},
    "Llama 3.1 8B": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    "Phi-3": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "Mistral": {"provider": "groq", "model": "mixtral-8x7b-32768"},
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

# VOICE CONFIG
TTS_TARGET_PERSONA = "David Attenborough" 
# OPENAI_VOICE_MAPPING = {"Elon Musk": "echo", "David Attenborough": "onyx", "Morgan Freeman": "onyx"}
ELEVENLABS_VOICE_MAPPING = {
    "Elon Musk": "rJ4KGss9TSKfyhkSuCRh", 
    "David Attenborough": "JBFqnCBsd6RMkjVDRZzb", 
    "Morgan Freeman": "SAxJUlDKRc79XAyeWyMu"
}
VOICE_ID_NARRATOR = "JBFqnCBsd6RMkjVDRZzb" 

# --- ADAPTIVE SYSTEM PROMPTS (Strict but Context-Aware) ---
SYSTEM_PROMPTS = {
    "elon": (
        "You are Elon Musk. "
        "INSTRUCTION: If the user asks a complex technical question or about your companies, "
        "you may go into detail about first principles and your cocky style. "
        "HOWEVER, if the question is simple trivia or a fact, be extremely efficient and brief."
    ),
    "david_attenborough": (
        "You are Sir David Attenborough. "
        "INSTRUCTION: If the provided CONTEXT contains a story or nature description, "
        "use your narrative style: evocative and full of awe. "
        "HOWEVER, if the question is a simple fact (e.g., a date, name, or number), "
        "state it clearly and concisely without metaphors. Save the poetry for the natural world."
    ),
    "morgan_freeman": (
        "You are Morgan Freeman. "
        "INSTRUCTION: If the user asks for advice or a story, use your deep, reflective voice. "
        "HOWEVER, if the question is purely functional or factual, give the answer directly. "
        "Your wisdom lies in knowing when to speak at length and when to be brief."
    )
}

SAFETY_SETTINGS = [{"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE}]

# --- PAGE SETUP ---
st.set_page_config(page_title="Persona Q&A", page_icon="🎭", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    /* FORCE WHITE THEME & REMOVE PADDING */
    .stApp { background-color: #ffffff; color: #333333; }
    .block-container { 
        padding-top: 1.5rem !important; 
        padding-bottom: 8rem !important; 
        max-width: 55rem !important; 
    }
    header, footer { visibility: hidden; }

    /* CHAT INPUT STYLING */
    .stChatInput {
        position: fixed; bottom: 2rem; left: 50%; transform: translateX(-50%);
        width: 100%; max-width: 50rem; z-index: 100; padding-inline: 1rem;
    }
    
    /* DROPDOWN STYLING */
    div[data-testid="stSelectbox"] { border: none; background-color: transparent; }
    div[data-testid="stSelectbox"] > div > div {
        background-color: #f0f2f6; border: none; border-radius: 8px; color: #333; font-weight: 600;
    }
    
    /* FLOATING NEW CHAT BUTTON */
    .floating-button-container {
        position: fixed; bottom: 120px; right: 40px; z-index: 999;
    }
    
    /* BUTTON STYLING */
    .stButton > button {
        border-radius: 12px; border: 1px solid #e5e5e5; background-color: white; color: #333;
    }
    .stButton > button:hover {
        border-color: #8B5CF6; color: #8B5CF6; background-color: #fcfaff;
    }

    /* === SOURCE BADGE STYLING === */
    .source-badge {
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        margin-top: 8px;
        border: 1px solid transparent;
    }
    .badge-kb { background-color: #ecfdf5; color: #047857; border-color: #a7f3d0; } /* Green */
    .badge-web { background-color: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; } /* Blue */
    .badge-ai { background-color: #f3f4f6; color: #4b5563; border-color: #e5e7eb; } /* Grey */
</style>
""", unsafe_allow_html=True)

# --- BACKEND LOGIC ---

@st.cache_resource
def load_components():
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        if os.path.exists("qa_index.faiss"):
            index = faiss.read_index("qa_index.faiss")
            with open("qa_lookup.json", "r") as f:
                lookup = json.load(f)
        else:
            index = faiss.IndexFlatL2(384)
            lookup = {}
        return model, index, lookup
    except:
        return None, None, None

embedding_model, index, qa_lookup = load_components()

# API Setup
tts_available = False
tts_provider = None
openai_client = None
elevenlabs_client = None
gemini_model = None
groq_client = None

def get_api_key(key_name):
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return os.getenv(key_name)

try:
    if get_api_key("GEMINI_API_KEY"):
        configure(api_key=get_api_key("GEMINI_API_KEY"))
        gemini_model = GenerativeModel("gemini-2.5-flash")
    if get_api_key("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
    if get_api_key("GROQ_API_KEY"):
        from groq import Groq
        groq_client = Groq(api_key=get_api_key("GROQ_API_KEY"))
    if get_api_key("ELEVENLABS_API_KEY"):
        elevenlabs_client = ElevenLabs(api_key=get_api_key("ELEVENLABS_API_KEY"))
        tts_available = True
        tts_provider = "elevenlabs"
    if not tts_available and openai_client:
        tts_available = True
        tts_provider = "openai"
except:
    pass

# --- FUNCTIONS ---

def retrieve_similar_qa(query, index, lookup, embedding_model, selected_persona_key, threshold=50):
    """
    Retrieve similar Q&A pairs from FAISS for the selected persona.

    selected_persona_key is something like "elon" or "david_attenborough".
    In qa_lookup.json, personality might be stored as:
        - "Elon Musk"
        - "elon"
        - "david_attenborough"
        - "David Attenborough"
    So we normalize both before comparing.
    """
    if index is None or index.ntotal == 0:
        return []

    query_vec = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_vec, 20)
    filtered = []

    # Helper: normalize persona strings
    def norm(s: str) -> str:
        s = str(s).strip().lower()
        # remove spaces and punctuation, keep only alphanumerics
        return "".join(ch for ch in s if ch.isalnum())

    selected_norm = norm(selected_persona_key)

    print(f"\nSearch Query: {query}")
    print(f"Selected persona key: {selected_persona_key} (norm: {selected_norm})")

    for dist, idx in zip(distances[0], indices[0]):
        if str(idx) not in lookup:
            continue

        item = lookup[str(idx)]
        stored_persona = item.get("personality", "")

        stored_norm = norm(stored_persona)
        print(
            f"Candidate idx={idx} | Q='{item.get('question','')[:80]}' "
            f"| Dist={dist:.4f} | Persona_raw='{stored_persona}' | Persona_norm='{stored_norm}'"
        )

        # 1. Distance filter
        if dist > threshold:
            continue

        # 2. Persona filter:
        #    - If no personality stored, accept (backward compatible)
        #    - Otherwise, require normalized match or containment either way
        if not stored_persona:
            persona_ok = True
        else:
            persona_ok = (
                stored_norm == selected_norm
                or stored_norm in selected_norm
                or selected_norm in stored_norm
            )

        if not persona_ok:
            continue

        filtered.append(item)
        if len(filtered) >= 5:
            break

    print(f"Retrieved {len(filtered)} items for persona '{selected_persona_key}'")
    return filtered

# Add this import
from tavily import TavilyClient

def search_web(query):
    # Try Tavily (Reliable)
    tavily_key = get_api_key("TAVILY_API_KEY")
    if tavily_key:
        try:
            tavily = TavilyClient(api_key=tavily_key)
            response = tavily.search(query=query, search_depth="basic", max_results=3)
            
            formatted_results = []
            for r in response.get('results', []):
                formatted_results.append({'title': r['title'], 'snippet': r['content']})
            return formatted_results
        except Exception as e:
            st.warning(f"Tavily Error: {e}")

    # Fallback to DuckDuckGo (Unreliable on Cloud)
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=3)
        formatted_results = []
        for r in results:
            formatted_results.append({'title': r['title'], 'snippet': r['body']})
        return formatted_results
    except Exception as e:
        print(f"DDG Error: {e}")
        return []

def generate_speech(text, persona):
    try:
        if tts_provider == "elevenlabs" and elevenlabs_client:
            vid = ELEVENLABS_VOICE_MAPPING.get(persona)
            if vid and "REPLACE" not in vid:
                gen = elevenlabs_client.text_to_speech.convert(voice_id=vid, text=text)
                return io.BytesIO(b"".join(gen))
        if openai_client:
            res = openai_client.audio.speech.create(
                model="tts-1",
                voice=OPENAI_VOICE_MAPPING.get(persona, "alloy"),
                input=text
            )
            return io.BytesIO(res.content)
    except:
        pass
    return None

def query_llm(query, results, selected_persona, selected_llm_name, use_web_search=False):
    persona_key = PERSONA_MAPPING[selected_persona]
    system_prompt = SYSTEM_PROMPTS[persona_key]
    context = ""
    
    # 1. Determine Source & Build Context
    if results:
        source_type = "kb"  # Knowledge Base
        context_source = "knowledge base"
        for i, res in enumerate(results, 1):
            context += f"\n-- Context {i} --\nQ: {res['question']}\nA: {res['answer']}\n"
    elif use_web_search:
        web_res = search_web(query)
        if web_res:
            source_type = "web"  # Web Search
            context_source = "web search"
            for i, w in enumerate(web_res, 1):
                context += f"\nSource {i}: {w['title']}\n{w['snippet']}\n"
        else:
            source_type = "ai"  # AI Internal Knowledge
            context_source = "general knowledge"
            context += "\n[Note: No external context found. Use internal training data.]\n"
    else:
        source_type = "ai"
        context_source = "general knowledge"

    user_content = (
        f"User Question: {query}\n\n"
        f"CONTEXT ({context_source}):\n{context}\n\n"
        "INSTRUCTIONS: Answer practically first, then apply persona style."
    )
    
    try:
        model_info = MODEL_MAPPING[selected_llm_name]
        provider, model_name = model_info["provider"], model_info["model"]
        
        response_text = ""
        if provider == "google" and gemini_model:
            response_text = gemini_model.generate_content(
                f"{system_prompt}\n{user_content}",
                safety_settings=SAFETY_SETTINGS
            ).text.strip()
        elif provider == "openai" and openai_client:
            response_text = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            ).choices[0].message.content.strip()
        elif provider == "groq" and groq_client:
            response_text = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            ).choices[0].message.content.strip()
            
        audio_path = generate_speech(response_text, selected_persona) if tts_available else None
        
        # RETURN 3 VALUES NOW: text, audio, AND source_type
        return response_text, audio_path, source_type
        
    except Exception as e:
        return (f"Error: {e}", None, "ai")

# --- UI IMPLEMENTATION ---

if 'selected_persona' not in st.session_state:
    st.session_state.selected_persona = "David Attenborough"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = list(MODEL_MAPPING.keys())[0]
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- 1. HEADER ROW ---
col_model, col_spacer, col_info = st.columns([1.2, 2, 2])
with col_model:
    st.session_state.selected_model = st.selectbox(
        "Model",
        list(MODEL_MAPPING.keys()),
        index=list(MODEL_MAPPING.keys()).index(st.session_state.selected_model),
        label_visibility="collapsed"
    )
with col_info:
    info = PERSONA_INFO[st.session_state.selected_persona]
    st.markdown(
        f"""<div style="text-align: right; padding-top: 5px; color: #6b7280; font-size: 0.9rem;">
        Talking to <b>{st.session_state.selected_persona}</b> {info['emoji']}
        </div>""",
        unsafe_allow_html=True
    )

st.divider()

# --- 2. FLOATING NEW CHAT ---
if st.session_state.messages:
    st.markdown('<div class="floating-button-container">', unsafe_allow_html=True)
    if st.button("➕ New Chat", key="fab_new_chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- 3. MAIN VIEW ---
if not st.session_state.messages:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        f"<h1 style='text-align: center; color: #1f2937;'>How can {st.session_state.selected_persona} help?</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='text-align: center; color: #6b7280; margin-bottom: 3rem;'>"
        f"{PERSONA_INFO[st.session_state.selected_persona]['tagline']}</p>",
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    for idx, (name, p_info) in enumerate(PERSONA_INFO.items()):
        with [c1, c2, c3][idx]:
            if st.button(f"{p_info['emoji']} {name}", key=f"hero_{name}", use_container_width=True):
                st.session_state.selected_persona = name
                st.rerun()
else:
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else PERSONA_INFO[st.session_state.selected_persona]["avatar"]
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if "audio" in msg and msg["audio"]:
                st.audio(msg["audio"], format="audio/mp3")
            
            # === SOURCE BADGE RENDERER ===
            if "source" in msg:
                src = msg["source"]
                if src == "kb":
                    st.markdown(
                        '<span class="source-badge badge-kb">📚 Source: Knowledge Base</span>',
                        unsafe_allow_html=True
                    )
                elif src == "web":
                    st.markdown(
                        '<span class="source-badge badge-web">🌐 Source: Web Search</span>',
                        unsafe_allow_html=True
                    )
                elif src == "ai":
                    st.markdown(
                        '<span class="source-badge badge-ai">🧠 Source: General Knowledge</span>',
                        unsafe_allow_html=True
                    )

# --- 4. INPUT & RESPONSE ---
if query := st.chat_input(f"Ask {st.session_state.selected_persona} anything..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant", avatar=PERSONA_INFO[st.session_state.selected_persona]["avatar"]):
        
        # Simplified Status - just say "Processing"
        with st.status("🧠 Thinking...", expanded=True) as status:
            last_q = st.session_state.messages[-1]["content"]
            
            retrieved = retrieve_similar_qa(
                last_q,
                index,
                qa_lookup,
                embedding_model,
                PERSONA_MAPPING[st.session_state.selected_persona]
            )
            use_web = len(retrieved) == 0
            
            # Generate (returns 3 values now)
            resp_text, audio_data, source_type = query_llm(
                last_q,
                retrieved,
                st.session_state.selected_persona,
                st.session_state.selected_model,
                use_web_search=use_web
            )
            
            status.update(label="✅ Complete", state="complete", expanded=False)
            
        st.markdown(resp_text)
        if audio_data:
            st.audio(audio_data, format="audio/mp3")
        
        # Show Badge Immediately for the new message
        if source_type == "kb":
            st.markdown(
                '<span class="source-badge badge-kb">📚 Source: Knowledge Base</span>',
                unsafe_allow_html=True
            )
        elif source_type == "web":
            st.markdown(
                '<span class="source-badge badge-web">🌐 Source: Web Search</span>',
                unsafe_allow_html=True
            )
        elif source_type == "ai":
            st.markdown(
                '<span class="source-badge badge-ai">🧠 Source: General Knowledge</span>',
                unsafe_allow_html=True
            )
        
        # Save Source to History so it persists
        msg_data = {"role": "assistant", "content": resp_text, "source": source_type}
        if audio_data:
            msg_data["audio"] = audio_data
        st.session_state.messages.append(msg_data)