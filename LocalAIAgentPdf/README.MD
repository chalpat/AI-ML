# PDF-based RAG System

## Overview

This project implements a **PDF-based Retrieval-Augmented Generation (RAG)** system.  
It reads input PDF files, converts the content into chunks, generates embeddings, and stores them in a **vector database**.  
When a user submits a query, the system retrieves **only the most relevant information from the loaded PDF(s)** and generates precise answers based strictly on that content.

---

## Features

- Load and process PDF documents
- Split PDF text into optimized chunks
- Generate embeddings for each chunk
- Store embeddings in a vector database
- Retrieve only relevant chunks for a query
- Generate accurate, PDF-specific answers
- Prevents hallucination by restricting responses to document context

---

## How It Works

1. PDF files are read and text is extracted
2. Extracted text is split into smaller chunks
3. Each chunk is converted into vector embeddings
4. Embeddings are stored in a vector database
5. User queries are embedded and matched against stored vectors
6. Only the most relevant PDF content is used to generate the response

---

## Technology Stack

- Python
- PDF parsing library (PyPDF / pdfplumber / similar)
- Embedding models (OpenAI, HuggingFace, or local models)
- Vector database (FAISS, Chroma, Pinecone, etc.)
- RAG framework (LangChain, LlamaIndex, or custom implementation)

---

## Project Structure

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

1. Place your PDF files in the `data/input_pdfs/` directory
2. Run the application:

   python app.py  

3. Ask questions related to the PDF content
4. The system responds using **only the loaded PDF data**

---

## Important Note

This RAG system **does not use external knowledge**.  
All responses are generated **strictly from the content of the provided PDF files**, ensuring accuracy and reliability.

---

## Use Cases

- PDF document question answering
- Research paper analysis
- Technical documentation assistants
- Policy and compliance document search
- Internal knowledge base systems

---

## Future Enhancements

- Support for additional file formats (DOCX, TXT, HTML)
- Metadata-based filtering
- Web or UI interface (Streamlit / Gradio)
- Multi-PDF comparison
- Performance and caching improvements

---

## Contributing

Contributions are welcome.  
Fork the repository, create a new branch, and submit a pull request with your improvements.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

Inspired by modern Retrieval-Augmented Generation (RAG) architectures and advancements in vector search and LLM-based systems.

---

⭐ If you find this project useful, consider starring the repository on GitHub!
