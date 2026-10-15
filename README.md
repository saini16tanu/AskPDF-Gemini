# 📄 AskPDF-Gemini

A fast, lightweight, and local Retrieval-Augmented Generation (RAG) assistant that allows you to chat with any PDF document using **Google Gemini**, **LangChain**, and **ChromaDB**.

---

## 🚀 Features

- **Document Ingestion**: Efficiently extracts and processes PDF pages using `PyPDFLoader`.
- **Smart Chunking**: Splits text using `RecursiveCharacterTextSplitter` with customizable chunk sizes and overlap to maintain contextual continuity.
- **Vector Embeddings**: Generates high-dimensional vector embeddings with Google's `models/gemini-embedding-001`.
- **Local Vector Store**: Persists document vectors locally with **ChromaDB** without requiring external cloud databases.
- **RAG Pipeline**: Built with LangChain Expression Language (LCEL) integrating retriever, prompt template, and Gemini (`gemini-3.1-flash-lite`).
- **Interactive & CLI Modes**: Supports interactive multi-turn terminal questioning or single-query CLI execution.
- **Resilient Rate Handling**: Built-in exponential backoff retry for Gemini API free-tier rate limits.

---

## 🛠️ Architecture

```
[ User PDF ] 
      │
      ▼ (PyPDFLoader)
[ Document Pages ]
      │
      ▼ (RecursiveCharacterTextSplitter)
[ Text Chunks ] 
      │
      ▼ (GoogleGenerativeAIEmbeddings)
[ Vector Embeddings ] ──▶ Stored in [ ChromaDB (Local /db) ]
                                          │
                                          ▼ (Similarity Search)
[ User Query ] ──▶ [ Top-k Context Chunks ] ──▶ [ Prompt Template ] ──▶ [ Gemini LLM ] ──▶ [ Grounded Answer ]
```

---

## 📋 Prerequisites

- Python 3.10+
- A Google Gemini API Key (Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey))

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/saini16tanu/AskPDF-Gemini.git
cd AskPDF-Gemini
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows use: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API Key
Copy `.env.example` to `.env` and insert your Gemini API Key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

---

## 📖 Usage

### Step 1: Ingest your PDF
Place your PDF into the project directory (e.g. `sample.pdf`) or pass any PDF file path:

```bash
# Ingest default sample.pdf
python ingest.py

# Or specify a custom PDF file
python ingest.py /path/to/your/document.pdf
```
This extracts text, computes embeddings, and stores them in the local `db/` folder.

### Step 2: Ask Questions

#### Option A: Single Query CLI
```bash
python ask.py "What is a tree and what is a binary tree?"
```

#### Option B: Interactive Terminal Session
```bash
python ask.py
```
```
============================================================
Ready! Ask questions about your ingested PDF.
Type your question and press Enter. Type 'exit' or 'q' to quit.
============================================================

Ask a question: What is the difference between a full and complete tree?
Searching and generating answer...

Answer:
The difference between a full binary tree and a complete binary tree is:
- Full binary tree: Every node has either 0 or 2 children.
- Complete binary tree: Every level is completely filled except possibly the last, which is filled from left to right.
```

---

## 📁 Project Structure

```
AskPDF-Gemini/
├── .env.example         # Example environment configuration
├── .gitignore           # Excludes virtualenv, .env secrets, and local Chroma db
├── ask.py               # RAG query engine & interactive QA terminal
├── ingest.py            # PDF loader, chunker, and vector store builder
├── requirements.txt     # Python package dependencies
├── pyrightconfig.json   # IDE / Language server configuration
├── sample.pdf           # Sample document for indexing
└── README.md            # Project documentation
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/saini16tanu/AskPDF-Gemini/issues).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
