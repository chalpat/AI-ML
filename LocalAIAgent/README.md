## Local RAG AI customized for reading a products csv and respond to specific queries related to price, etc.

### 📌  Overview

This project implements a Local Retrieval-Augmented Generation (RAG) AI system designed to read and understand a product dataset stored in a CSV file. It enables users to ask natural language questions and receive accurate, context-aware responses related to product details such as price, availability, category, and other attributes.

The system runs locally, ensuring better data privacy, lower latency, and full control over the data and models used.

### 🚀 Features

📊 Reads and indexes product data from a CSV file

🔍 Retrieval-Augmented Generation (RAG) for precise answers

💬 Natural language query support (e.g., “What is the price of Product X?”)

🧠 Embedding-based semantic search for relevant rows

🏠 Fully local execution (no mandatory cloud dependency)

🔧 Easily customizable for different CSV schemas

### 🛠️ Tech Stack

Python

LLM (local or API-based, configurable)

Embedding model (HuggingFace / OpenAI / Local)

Vector store (FAISS / Chroma / equivalent)

# Local RAG AI – Product CSV Query Assistant

## 📖 Description
This project is a **Local Retrieval-Augmented Generation (RAG) AI system** customized to read a **products CSV file** and respond intelligently to **specific user queries** such as product price, category, availability, and other related attributes.

By combining semantic search with a Large Language Model (LLM), the system retrieves relevant product records from the CSV and generates accurate, context-aware answers — all while running **locally** for improved privacy and control.

---

## ✨ Key Features
- 📊 Reads and processes product data from a CSV file  
- 🔍 Semantic search using vector embeddings  
- 🤖 Retrieval-Augmented Generation (RAG) based responses  
- 💬 Natural language queries (e.g., “What is the price of Product X?”)  
- 🏠 Fully local execution (no mandatory cloud dependency)  
- 🔧 Easily customizable for different CSV schemas  

---

## 🧠 How It Works
1. Product data is loaded from a CSV file  
2. Each row is converted into embeddings and stored in a vector index  
3. User queries are matched semantically against indexed data  
4. The LLM generates responses using the retrieved context  

---

## 🛠️ Tech Stack
- Python  
- Local or API-based LLM (configurable)  
- Embedding models (HuggingFace / OpenAI / Local)  
- Vector database (FAISS / ChromaDB / similar)  

---

## 📂 Project Structure
├── data/
│ └── products.csv
├── src/
│ ├── ingest.py # CSV ingestion and vector indexing
│ ├── rag_engine.py # RAG pipeline implementation
│ └── query.py # Query interface
├── requirements.txt
└── README.md

### 📥 Installation

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt

### ▶️ Usage

Place your product CSV file inside the data/ directory.

Run the ingestion process to build the vector index:

python src/ingest.py

Start querying the system:

python src/query.py

### 🧪 Example Queries

“What is the price of Product A?”

“List products under ₹10,000”

“Which product has the highest rating?”

“Show all electronics products in stock”

### 🔧 Customization

Modify CSV column mappings in the ingestion script

Swap embedding or LLM models based on performance needs

Extend response logic for analytics or summaries

### 🔐 Data Privacy

All data processing and inference can be performed locally, making this solution ideal for sensitive or proprietary product information.

### 📌 Use Cases

Product catalog search

Internal pricing intelligence

E-commerce analytics

Inventory & sales support tools

### 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
