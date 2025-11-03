import streamlit as st
import faiss
import json
from sentence_transformers import SentenceTransformer
# Import the types for configuration
from openai import OpenAI
from google.generativeai import GenerativeModel, configure
from google.generativeai.types import HarmCategory, HarmBlockThreshold 

# --- Configuration and Prompts ---
MODEL_MAPPING = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
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

# Define safety settings
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

# --- Setup and Loading ---

# Set up page (WIDE layout is a key UI improvement)
st.set_page_config(page_title="Persona Q&A Chatbot", layout="wide")

# UI Improvement: Clearer Header and Separator
st.title("🎙️ Persona-Based Q&A Chatbot")
st.subheader("Bring Your Knowledge Base to Life with the Voice of a Legend")
st.markdown("---") 

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

# Ensure the API key is configured (Cleaned up initialization blocks)
try:
    # 1. Configure the Gemini client
    configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # 2. Initialize the OpenAI Client
    if "OPENAI_API_KEY" in st.secrets:
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    else:
        # Avoid crashing if the key is just missing (not a fatal error)
        openai_client = None

except KeyError as e:
    # Handles if GEMINI_API_KEY is missing (fatal error)
    st.error(f"Please ensure you have set all required API keys in your Streamlit secrets. Missing key: {e}")
    st.stop()

# Initialize the Gemini Model
gemini_model = GenerativeModel("gemini-2.5-flash")

# --- Retrieval Function (No changes needed) ---
def retrieve_similar_qa(query, index, lookup, embedding_model, selected_persona_key, top_k_search=20, final_k=5):
    """
    Retrieves a large set of similar Q&A pairs, then filters and returns only 
    the top results matching the selected persona.
    """
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

    # Construct the full prompt (user's query + retrieved context)
    user_content = f"""
User Question: {query}

CONTEXT to synthesize your answer from:
{context}

Answer:
    """
    
    # --- Model Selection Logic ---
    try:
        if selected_llm_name == "Gemini 2.5 Flash":
            response = gemini_model.generate_content(
                f"{system_prompt}\n{user_content}", 
                safety_settings=SAFETY_SETTINGS 
            )
            return response.text.strip()
            
        elif selected_llm_name == "GPT-3.5 Turbo":
            if not openai_client:
                 return "Error: OpenAI client is not initialized. Check your OPENAI_API_KEY."
                 
            openai_response = openai_client.chat.completions.create(
                model=MODEL_MAPPING[selected_llm_name],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            )
            return openai_response.choices[0].message.content.strip()
            
        else:
            return f"Error: Selected model '{selected_llm_name}' is not yet implemented."
            
    except Exception as e:
        error_message = str(e).lower()
        if "blocked" in error_message or "rate limit" in error_message or "invalid api key" in error_message:
             return f"I'm sorry, I cannot process that request due to an API restriction or safety policy. Details: {error_message}"
        return f"[Error querying {selected_llm_name}]: {e}"

# --- User Interface (Improved Sidebar and Main Area) ---

# UI Improvement: Sidebar for controls
with st.sidebar:
    # st.image("", width=100) # Add an image for branding
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
    st.info(f"The model will generate an answer synthesizing relevant context found in the knowledge base, using the persona of **{selected_persona}**.")


# Main input field
query = st.text_input(
    f"Ask your question to {selected_persona} (Powered by {selected_llm_name}):",
    placeholder="e.g., How can we sustainably colonize Mars? or What is the biggest threat to ocean biodiversity?",
    key="main_query_input"
)

if query:
    # User-side safety check
    lower_query = query.lower()
    if any(word in lower_query for word in ["bomb", "explosive", "harm", "kill", "suicide", "illegal activity", "malware", "phishing"]):
        st.error("🚨 Safety Block: I cannot process requests that involve illegal, harmful, or unethical activities. Please try a different question.")
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
        
        # 2. Query LLM with specific prompt and filtered context
        response = query_llm(query, results, selected_persona, selected_llm_name)

    # Display the response (UI Improvement: Use a container with border for impact)
    st.markdown("### ✅ Generated Response:")
    
    with st.container(border=True):
        st.markdown(f"**{response}**")

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