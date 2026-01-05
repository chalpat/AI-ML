# LangChain + OpenAI + LangSmith Integration 🚀

![LangChain + LangSmith](https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/langchain_stack.png)

## 📌 Project Overview

This project demonstrates **LangChain integration with OpenAI models and LangSmith** for building, tracing, debugging, and monitoring LLM-powered applications.

LangSmith is used to:
- Trace LangChain executions
- Debug prompt chains
- Monitor latency, token usage, and errors
- Improve prompt and chain reliability

The project serves as a **hands-on reference** for developers who want observability and evaluation for their LangChain-based AI workflows.

---

## 🧩 Architecture Overview

![LangSmith Architecture](https://docs.smith.langchain.com/assets/images/overview-diagram-9f1f3b2a4c9e1b3f5d6d2a6a4e9c8b2f.png)

**Flow:**
1. User sends a query
2. LangChain processes prompts and chains
3. OpenAI model generates responses
4. LangSmith captures traces, metadata, and performance metrics

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **LangChain**
- **OpenAI (Chat Models)**
- **LangSmith**
- **dotenv** (Environment variable management)

---

## 📂 Project Structure

```text
.
├── app.py                # Main LangChain + LangSmith integration
├── requirements.txt      # Project dependencies
├── .env                  # API keys and configuration
├── README.md             # Project documentation
