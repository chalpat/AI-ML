# PDF-based RAG System

## 📌 Overview

This project implements a **PDF-based Retrieval-Augmented Generation (RAG)** system.  
It reads input PDF files, converts the content into chunks, generates embeddings, and stores them in a **vector database**.  
When a user submits a query, the system retrieves **only the most relevant information from the loaded PDF(s)** and generates accurate, context-specific answers.

---

## 🚀 Architecture Diagram

**Flow Explanation:**
1. PDF files are ingested
2. Text is chunked into smaller segments
3. Embeddings are generated for each chunk
4. Vectors are stored in a vector database
5. User queries retrieve relevant chunks
6. LLM generates answers strictly from retrieved context

---

## 🚀 Features

- 📄 Load and parse PDF documents
- ✂️ Intelligent text chunking
- 🧠 Embedding generation
- 🗄️ Vector database storage
- 🔍 Context-aware retrieval
- 💬 Accurate PDF-only responses
- 🚫 No hallucinations or external knowledge

---

## 🛠️ System Workflow

---

## 🚀 How It Works

1. **PDF Ingestion**  
   Extracts raw text from PDF files.

2. **Chunking**  
   Breaks text into overlapping chunks for better semantic retrieval.

3. **Embedding Generation**  
   Converts chunks into dense vector representations.

4. **Vector Storage**  
   Stores embeddings in a vector database for fast similarity search.

5. **Query Processing**  
   User queries are embedded and matched against stored vectors.

6. **Answer Generation**  
   Responses are generated only from the retrieved PDF chunks.

---

## 🛠️ Technology Stack



- Python
- PDF loaders (PyPDF / pdfplumber)
- Embedding models (OpenAI / HuggingFace / local models)
- Vector databases (FAISS / Chroma / Pinecone)
- RAG frameworks (LangChain / LlamaIndex)

---

## 📂 Project Structure

.
├── data/  
│   └── input_pdfs/  
├── embeddings/  
├── vector_db/  
├── src/  
│   ├── pdf_loader.py  
│   ├── chunker.py  
│   ├── embedder.py  
│   ├── vector_store.py  
│   └── rag_pipeline.py  
├── app.py  
├── requirements.txt  
└── README.md  

---

## Installation

1. Clone the repository:

   git clone https://github.com/your-username/pdf-rag-system.git  
   cd pdf-rag-system  

2. Create and activate a virtual environment:

   python -m venv venv  
   source venv/bin/activate  
   (On Windows: venv\Scripts\activate)

3. Install dependencies:

   pip install -r requirements.txt  

---

## Usage

![Usage Example](assets/usage_example.png)

1. Place PDF files inside `data/input_pdfs/`
2. Run the application:

   python app.py  

3. Ask questions related to the PDF content
4. Receive answers strictly derived from the loaded documents

---

## Important Note

> ⚠️ This RAG system **does not use external knowledge**.  
> All responses are generated **only from the content of the provided PDF files**, ensuring reliability and correctness.

---

## Use Cases

![Use Cases](assets/use_cases.png)

- PDF question answering
- Research paper analysis
- Technical manuals and guides
- Policy and compliance documents
- Internal enterprise knowledge bases

---

## Future Enhancements

- Support for DOCX, TXT, and HTML files
- Metadata-based filtering
- UI integration (Streamlit / Gradio)
- Multi-document comparison
- Performance optimization and caching

---

## Contributing

Contributions are welcome.  
Fork the repository, create a feature branch, and submit a pull request.

---

## License

This project is licensed under the **MIT License**.

---

## Acknowledgements

Inspired by modern **Retrieval-Augmented Generation (RAG)** architectures and advancements in vector search and LLM-powered systems.

---

⭐ If you find this project useful, please consider starring the repository on GitHub!
