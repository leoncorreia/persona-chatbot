import streamlit as st
import faiss
import json
from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel, configure

# --- Configuration and Prompts ---

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
    # Use st.secrets for secure key management in Streamlit Cloud
    configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("Please ensure you have set GEMINI_API_KEY in your Streamlit secrets.")
    st.stop()

gemini_model = GenerativeModel("gemini-2.5-flash")

# --- Retrieval Function (Modified) ---

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

# --- Query Gemini Function (Modified) ---

def query_llm(query, results, selected_persona):
    if not results:
        # Fallback if no relevant documents were found for the persona
        return f"I am {selected_persona}, and unfortunately, I could not find specific context in my knowledge base to answer your question: '{query}'. Perhaps you could ask me about something else."

    # Get the dedicated system prompt
    persona_key = PERSONA_MAPPING[selected_persona]
    system_prompt = SYSTEM_PROMPTS[persona_key]

    context = ""
    for i, res in enumerate(results, 1):
        # We only include the Answer and Question in the final context block
        context += f"\n-- Retrieved Context {i} --\n"
        context += f"Q: {res['question']}\n"
        context += f"A: {res['answer']}\n"

    # 3. Use the persona-specific system prompt to guide the model
    prompt = f"""
{system_prompt}

User Question: {query}

CONTEXT to synthesize your answer from:
{context}

Answer:
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Error querying Gemini]: {e}"

# --- User Interface (Modified) ---

# UI for Persona Selection and Input
with st.sidebar:
    st.header("1. Select Your Guide")
    persona_options = list(PERSONA_MAPPING.keys())
    selected_persona = st.selectbox(
        "Choose a persona to answer your question:",
        persona_options
    )
    
    st.header("2. Ask a Question")
    # Store the persona key for filtering the index
    selected_persona_key = PERSONA_MAPPING[selected_persona]

# Main input field
query = st.text_input(f"Ask your question to {selected_persona}:")

if query:
    with st.spinner(f"Retrieving knowledge for {selected_persona} and generating response..."):
        
        # 1. Retrieve and Filter based on persona
        results = retrieve_similar_qa(
            query, 
            index, 
            qa_lookup, 
            embedding_model, 
            selected_persona_key
        )
        
        # 2. Query LLM with specific prompt and filtered context
        response = query_llm(query, results, selected_persona)

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