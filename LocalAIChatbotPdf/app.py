from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from dotenv import load_dotenv
import os
import streamlit as st

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Local AI PDF Chatbot", layout="wide")
st.title("📄 Local AI Chatbot for PDF Documents")

input_text = st.text_input("Enter your question here:")

# -----------------------------
# Cache-heavy operations
# -----------------------------
@st.cache_resource
def load_index():
    # Load documents
    documents = SimpleDirectoryReader(
        r"C:\\Users\\Chalpat Rauth\\Documents\\vsc_workspace\\AI-ML\\LocalAIChatbotPdf\\data"
    ).load_data()

    # LLM
    llm = OpenAI(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=256
    )

    # Embeddings (UNCHANGED)
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Global Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 50

    # Build index
    index = VectorStoreIndex.from_documents(documents)
    return index

# Load cached index
index = load_index()

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=3)

# -----------------------------
# Query from UI
# -----------------------------
if input_text:
    with st.spinner("Searching documents..."):
        response = query_engine.query(input_text)
        st.write(response.response)
