# 🔥 RAG + Groq + Langfuse Project

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using:

* FAISS (vector database)
* HuggingFace embeddings
* Groq (LLaMA 3.1 LLM)
* Langfuse (observability)

## 🧠 Architecture

1. Load documents from `data.txt`
2. Split into chunks
3. Convert into embeddings
4. Store in FAISS
5. Retrieve relevant chunks
6. Generate answer using Groq LLM
7. Track interactions with Langfuse

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

Create `.env` file:

```
GROQ_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_secret
LANGFUSE_HOST=https://cloud.langfuse.com
```

## ▶️ Run

```bash
python main.py
```

## 💡 Example

```
Question: What is RAG?
Answer: RAG combines retrieval with generation for better answers.
```

## 🔥 Key Features

* Reduces hallucination using RAG
* Fast inference using Groq
* Observability with Langfuse
* Modular and scalable architecture

## 📌 Future Improvements

* Streamlit UI
* PDF ingestion
* Chat memory
* Deployment

---

⭐ If you like this project, give it a star!
