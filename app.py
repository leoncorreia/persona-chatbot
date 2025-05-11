import streamlit as st
import faiss
import json
from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel, configure

# Set up page
st.set_page_config(page_title="Persona Q&A Chatbot", layout="wide")
st.title("🎙️ Persona-Based Q&A Chatbot")

# Load everything
@st.cache_resource
def load_components():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    index = faiss.read_index("qa_index.faiss")
    with open("qa_lookup.json", "r", encoding="utf-8") as f:
        lookup = json.load(f)
    return model, index, lookup

embedding_model, index, qa_lookup = load_components()
configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = GenerativeModel("gemini-1.5-flash-002")

# Retrieval
def retrieve_similar_qa(query, index, lookup, embedding_model, top_k=5):
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        if str(idx) in lookup:
            results.append(lookup[str(idx)])
    return results

# Query Gemini
def query_llm(query, results):
    if not results:
        return "No relevant answers found."

    context = ""
    for i, res in enumerate(results, 1):
        context += f"\nResult {i}:\n"
        context += f"Personality: {res['personality']}\n"
        context += f"Question: {res['question']}\n"
        context += f"Answer: {res['answer']}\n"
        context += f"Metadata: {res['metadata']}\n"

    prompt = f"""
You are a chatbot simulating the style of the person described in the context. Use the following Q&A context to answer the user's question in that personality's style.

User Question: {query}

Context: {context}

Answer:
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Error querying Gemini]: {e}"

# UI
query = st.text_input("Ask your question:")

if query:
    with st.spinner("Thinking..."):
        results = retrieve_similar_qa(query, index, qa_lookup, embedding_model)
        response = query_llm(query, results)

    st.markdown("### 💬 Gemini Response:")
    st.success(response)

    with st.expander("🧠 Show Retrieved Q&A Context"):
        for res in results:
            st.markdown(f"**Personality:** {res['personality']}")
            st.markdown(f"**Q:** {res['question']}")
            st.markdown(f"**A:** {res['answer']}")
            st.markdown(f"**Metadata:** `{res['metadata']}`")
            st.markdown("---")
