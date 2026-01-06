# 📄 LocalAIAgentPdf – Streamlit UI Integration

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![RAG](https://img.shields.io/badge/RAG-PDF%20Querying-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🚀 Overview

**LocalAIAgentPdf – Streamlit Integration** is a user-friendly web application built using **Streamlit** that enables **dynamic querying of PDF documents** loaded into the **LocalAIAgentPdf RAG system**.

This project provides an intuitive UI layer on top of your existing **PDF-based Retrieval-Augmented Generation (RAG)** pipeline, allowing users to:
- Upload PDFs
- Ask natural language questions
- Receive **context-aware, document-specific answers**

---

## 🖼️ Application Preview

### 🔹 Home Screen
### 🔹 PDF Upload & Indexing
### 🔹 Querying the PDF
### 🔹 Answer with Source Context

---

## 🧠 Architecture

```text
PDF Document
     ↓
Text Chunking
     ↓
Embeddings Generation
     ↓
Vector Database
     ↓
Retriever
     ↓
LLM
     ↓
Streamlit UI Response


