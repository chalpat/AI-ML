from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load documents
documents = SimpleDirectoryReader(
    r"C:\\Users\\Chalpat Rauth\\Documents\\vsc_workspace\\LocalAIAgentPdf\\data"
).load_data()

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

# Use an open, small chat model (NOTE: Adjust model name as needed for good responses)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

# Optionally move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# Global settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024
Settings.chunk_overlap = 50

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("How many CoEs are there in the document?")
print(response)