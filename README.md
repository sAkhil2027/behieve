# 📄 PDF RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit**, **FAISS**, **LangChain**, **HuggingFace Embeddings**, and **Groq LLMs**.

Upload any PDF document and chat with it in natural language. The application extracts text, generates embeddings, retrieves the most relevant content using semantic search, and produces grounded answers using a Large Language Model.

---

## 🌟 Overview

Large Language Models are powerful, but they cannot answer questions about private documents they have never seen before.

This project solves that problem using **Retrieval-Augmented Generation (RAG)**:

1. Upload a PDF.
2. Extract text from the document.
3. Split the text into semantic chunks.
4. Convert chunks into vector embeddings.
5. Store vectors in FAISS.
6. Retrieve relevant chunks for a user's query.
7. Send retrieved context to the LLM.
8. Generate accurate, context-aware answers.

---

## ✨ Features

* 📄 PDF Upload Support
* 🔍 Semantic Search using Vector Embeddings
* ⚡ Fast Retrieval with FAISS
* 🤖 Groq-powered LLM Responses
* 💬 Interactive Chat Interface
* 🧠 Context-Aware Question Answering
* 🔎 Retrieved Chunk Inspection
* 🔄 Streaming Responses
* 🗑️ Chat Reset Functionality
* 🎯 Hallucination Reduction through RAG

---

## 🏗️ System Architecture

```text
                ┌─────────────────┐
                │   Upload PDF    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Text Extraction │
                │     (PyPDF)     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Text Chunking   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ HuggingFace     │
                │ Embeddings      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ FAISS Vector DB │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Similarity      │
                │ Search          │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Retrieved       │
                │ Context         │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Groq LLM        │
                │ (Llama 3.3)     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Final Answer    │
                └─────────────────┘
```

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### LLM

* Groq API
* Llama 3.3 70B Versatile

### Retrieval Pipeline

* LangChain
* FAISS

### Embeddings

* HuggingFace Sentence Transformers
* all-MiniLM-L6-v2

### Document Processing

* PyPDF

### Environment Management

* Python Dotenv

---

## 📂 Project Structure

```bash
PDF-RAG-Chatbot/
│
├── app.py
├── .env
├── requirements.txt
├── README.md
│
└── assets/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / MacOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Application will start on:

```text
http://localhost:8501
```

---

## 📖 How It Works

### Step 1: Upload PDF

The user uploads a PDF document through the Streamlit interface.

---

### Step 2: Text Extraction

The application extracts text from all pages using:

```python
PdfReader()
```

---

### Step 3: Chunking

The extracted text is divided into smaller overlapping chunks.

```python
chunk_size = 1000
chunk_overlap = 200
```

This helps preserve context while improving retrieval quality.

---

### Step 4: Embedding Generation

Each chunk is converted into dense vector embeddings using:

```python
sentence-transformers/all-MiniLM-L6-v2
```

---

### Step 5: Vector Storage

Embeddings are stored inside a FAISS index for efficient similarity search.

---

### Step 6: Retrieval

For every question:

```python
similarity_search(query, k=4)
```

retrieves the top relevant chunks.

---

### Step 7: Context Injection

Retrieved chunks are combined into a context window and inserted into the system prompt.

---

### Step 8: Response Generation

The context is passed to:

```text
llama-3.3-70b-versatile
```

running on Groq infrastructure.

The model generates an answer grounded only in the retrieved content.

---

## 🔒 Hallucination Prevention

The assistant is instructed to answer **only from retrieved context**.

If the information is unavailable, it responds:

```text
I couldn't find this information in the PDF.
```

This significantly reduces hallucinations and improves reliability.

---

## 📊 Current Limitations

* Works primarily with text-based PDFs.
* Scanned PDFs require OCR support.
* Vector database exists only during runtime.
* Single-document processing.
* No document persistence after restart.

---

## 👨‍💻 Author

**Akhil Vikram Singh**

AI Engineer | Machine Learning Enthusiast | RAG & Agentic AI Developer

---

## ⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork the project

📢 Share it with others

---
