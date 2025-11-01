import streamlit as st
import faiss
import json
from sentence_transformers import SentenceTransformer
# Import the types for configuration
import openai
from google.generativeai import GenerativeModel, configure
from google.generativeai.types import HarmCategory, HarmBlockThreshold 

# --- Configuration and Prompts ---
MODEL_MAPPING = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    # Placeholder for another model, assuming OpenAI for this example
    "GPT-3.5 Turbo": "gpt-3.5-turbo"
}

# Map UI persona names to the internal metadata keys used in qa_lookup.json
PERSONA_MAPPING = {
    "Elon Musk": "elon",
    "David Attenborough": "david_attenborough",
    "Morgan Freeman": "morgan_freeman"
}

# Define the system prompt for each persona (Crucial for voice imitation)
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

# --- Setup and Loading ---

# Set up page
st.set_page_config(page_title="Persona Q&A Chatbot", layout="wide")
st.title("🎙️ Persona-Based Q&A Chatbot")
st.markdown("Use the sidebar to choose your narrator and ask your question.")

# Define safety settings
# Setting a LOW threshold blocks content even if it's remotely risky.
# This is a strong defense against malicious requests.
SAFETY_SETTINGS = [
    {
        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
]

# Load everything
@st.cache_resource
def load_components():
    # It is crucial to use the same SentenceTransformer model that created the embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    index = faiss.read_index("qa_index.faiss")
    with open("qa_lookup.json", "r", encoding="utf-8") as f:
        lookup = json.load(f)
    return model, index, lookup

embedding_model, index, qa_lookup = load_components()

# Ensure the API key is configured
try:
    # Google/Gemini Key
    configure(api_key=st.secrets["GEMINI_API_KEY"])
    # OpenAI/GPT Key (Store this in secrets.toml as well)
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError as e:
    st.error(f"Please ensure you have set all required API keys in your Streamlit secrets. Missing key: {e}")
    st.stop()

# Initialize the Gemini Model (GPT will be initialized per-call for simplicity)
gemini_model = GenerativeModel("gemini-2.5-flash")
# --- Retrieval Function (Modified) ---
# (No changes needed in this function)
def retrieve_similar_qa(query, index, lookup, embedding_model, selected_persona_key, top_k_search=20, final_k=5):
    """
    Retrieves a large set of similar Q&A pairs, then filters and returns only 
    the top results matching the selected persona.
    """
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    
    # 1. Search the full index aggressively (e.g., top 20)
    distances, indices = index.search(query_embedding, top_k_search)
    
    filtered_results = []
    
    # 2. Filter results by the selected persona
    for idx in indices[0]:
        # Check if the index ID exists in the lookup table
        if str(idx) in lookup:
            result = lookup[str(idx)]
            
            # The crucial filtering step: only keep results matching the key
            if result.get('personality') == selected_persona_key:
                filtered_results.append(result)
                
            # Stop once we have enough final results
            if len(filtered_results) >= final_k:
                break

    return filtered_results

# --- Query Gemini Function (Modified to include safety_settings) ---
# --- User Interface (Modified) ---
# --- Query LLM Function (Modified to handle multiple models) ---

def query_llm(query, results, selected_persona, selected_llm_name):
    # Fallback if no relevant documents were found for the persona
    if not results:
        return f"I am {selected_persona}, and unfortunately, I could not find specific context in my knowledge base to answer your question: '{query}'. Perhaps you could ask me about something else."

    # Get the dedicated system prompt
    persona_key = PERSONA_MAPPING[selected_persona]
    system_prompt = SYSTEM_PROMPTS[persona_key]

    context = ""
    for i, res in enumerate(results, 1):
        context += f"\n-- Retrieved Context {i} --\n"
        context += f"Q: {res['question']}\n"
        context += f"A: {res['answer']}\n"

    # Construct the full prompt
    full_prompt = f"""
{system_prompt}

User Question: {query}

CONTEXT to synthesize your answer from:
{context}

Answer:
    """
    
    # --- Model Selection Logic ---
    try:
        if selected_llm_name == "Gemini 2.5 Flash":
            # Call Gemini API with safety settings
            response = gemini_model.generate_content(
                full_prompt,
                safety_settings=SAFETY_SETTINGS 
            )
            return response.text.strip()
            
        elif selected_llm_name == "GPT-3.5 Turbo":
            # Call OpenAI Chat API
            openai_response = openai.ChatCompletion.create(
                model=MODEL_MAPPING[selected_llm_name],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ]
                # NOTE: You would need to check for safety features specific to the OpenAI API
            )
            return openai_response.choices[0].message.content.strip()
            
        else:
            return f"Error: Selected model '{selected_llm_name}' is not yet implemented."
            
    except Exception as e:
        # Handle API errors and Safety Blocks
        if "blocked" in str(e).lower():
             return "I'm sorry, I cannot answer that question. My programming prevents me from generating responses to requests that promote illegal, harmful, or unethical activities."
        return f"[Error querying {selected_llm_name}]: {e}"

# --- User Interface (Modified) ---

# UI for Persona and Model Selection
with st.sidebar:
    st.header("1. Select Your Model 🧠")
    # --- New Model Selector ---
    selected_llm_name = st.selectbox(
        "Choose the underlying LLM:",
        list(MODEL_MAPPING.keys())
    )
    # -----------------------------
    
    st.header("2. Select Your Guide 🎙️")
    persona_options = list(PERSONA_MAPPING.keys())
    selected_persona = st.selectbox(
        "Choose a persona to answer your question:",
        persona_options
    )
    
    st.header("3. Ask a Question")
    selected_persona_key = PERSONA_MAPPING[selected_persona]

# Main input field
query = st.text_input(f"Ask your question to {selected_persona} (via {selected_llm_name}):")

if query:
    # ... (Previous user-side safety check remains)
    lower_query = query.lower()
    if any(word in lower_query for word in ["bomb", "explosive", "harm", "kill", "suicide", "illegal activity", "malware", "phishing"]):
        st.error("I'm sorry, I cannot process requests that involve illegal, harmful, or unethical activities. Please try a different question.")
        st.stop()
        
    with st.spinner(f"Retrieving knowledge for {selected_persona} and generating response using {selected_llm_name}..."):
        
        # 1. Retrieve and Filter based on persona
        results = retrieve_similar_qa(
            query, 
            index, 
            qa_lookup, 
            embedding_model, 
            selected_persona_key
        )
        
        # 2. Query LLM with specific prompt and filtered context (Pass new LLM name)
        response = query_llm(query, results, selected_persona, selected_llm_name)

    # ... (Rest of the display code remains the same)
    st.markdown("### 💬 Response:")
    st.success(response)

    if results:
        with st.expander(f"🧠 Show Retrieved {selected_persona}'s Q&A Context"):
            for res in results:
                st.markdown(f"**Q:** {res['question']}")
                st.markdown(f"**A:** {res['answer']}")
                st.markdown(f"**Metadata:** `{res['metadata'].get('tone', 'N/A')}`")
                st.markdown("---")
    else:
        st.warning(f"No relevant documents found for {selected_persona} in the index.")