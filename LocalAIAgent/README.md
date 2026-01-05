### Local RAG AI customized for reading a products csv and respond to specific queries related to price, etc.

📌 #### Overview

This project implements a Local Retrieval-Augmented Generation (RAG) AI system designed to read and understand a product dataset stored in a CSV file. It enables users to ask natural language questions and receive accurate, context-aware responses related to product details such as price, availability, category, and other attributes.

The system runs locally, ensuring better data privacy, lower latency, and full control over the data and models used.

🚀 #### Features

📊 Reads and indexes product data from a CSV file

🔍 Retrieval-Augmented Generation (RAG) for precise answers

💬 Natural language query support (e.g., “What is the price of Product X?”)

🧠 Embedding-based semantic search for relevant rows

🏠 Fully local execution (no mandatory cloud dependency)

🔧 Easily customizable for different CSV schemas

🛠️ Tech Stack

Python

LLM (local or API-based, configurable)

Embedding model (HuggingFace / OpenAI / Local)

Vector store (FAISS / Chroma / equivalent)

📂 Project Structure
├── data/
│   └── products.csv
├── src/
│   ├── ingest.py
│   ├── rag_pipeline.py
│   └── query.py
├── requirements.txt
└── README.md

📥 Installation
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt

▶️ Usage

Place your product CSV file inside the data/ directory.

Run the ingestion process to build the vector index:

CSV-based data ingestion
