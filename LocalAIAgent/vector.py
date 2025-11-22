from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# CSV with your product data
df = pd.read_csv("products.csv")

# Embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Directory where Chroma will store vectors
db_location = "./chroma_langchain_db"

# Check if this is the first run (no existing DB)
add_documents = not os.path.exists(db_location)

# Always initialize the vectorstore
vectorstore = Chroma(
    collection_name="products_collection",
    embedding_function=embeddings,
    persist_directory=db_location,
)

# Only add documents on first run
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # You can also include Description/Brand, etc. here if you want better semantic search
        page_content = f"Name: {row['Name']}\nCategory: {row['Category']}\nPrice: {row['Price']} {row['Currency']}"
        document = Document(
            page_content=page_content,
            metadata={
                "price": int(row["Price"]),
                "name": row["Name"],
                "category": row["Category"],
                "currency": row["Currency"],
            },
        )
        ids.append(str(row["Index"]))
        documents.append(document)

    vectorstore.add_documents(documents=documents, ids=ids)

# Export a retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
