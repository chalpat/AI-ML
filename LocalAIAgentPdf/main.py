from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Ensure OpenAI key is set
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Load documents
# -----------------------------
documents = SimpleDirectoryReader(
    r"C:\\Users\\Chalpat Rauth\\Documents\\vsc_workspace\\LocalAIAgentPdf\\data"
).load_data()

# -----------------------------
# Prompt Template
# -----------------------------
qa_template = PromptTemplate(
    """
You are a careful assistant.

Answer the question using ONLY the information in the context below.
If the answer is not clearly in the context, say:
"I don't know based on the provided document."

Do not use any outside knowledge. Do not guess.

Context:
{context_str}

Question:
{query_str}

Answer:
"""
)

# -----------------------------
# OpenAI LLM (Latest)
# -----------------------------
llm = OpenAI(
    model="gpt-4.1",   # alternatives: gpt-4o, gpt-4.1, gpt-4.1-mini
    temperature=0.0,
    max_tokens=256
)

# -----------------------------
# Embedding Model
# -----------------------------
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# -----------------------------
# Global Settings
# -----------------------------
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024
Settings.chunk_overlap = 50
Settings.prompt_template = qa_template

# -----------------------------
# Build Index
# -----------------------------
index = VectorStoreIndex.from_documents(documents)

# -----------------------------
# Query Engine
# -----------------------------
query_engine = index.as_query_engine()
response = query_engine.query("How many CoEs are there in the document?")
#response = query_engine.query("How is India doing today?")

print(response)
