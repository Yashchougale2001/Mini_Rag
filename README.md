Mini-RAG: CPU-Friendly Retrieval-Augmented Generation System

A lightweight, modular Retrieval-Augmented Generation (RAG) system designed specifically for CPU-only environments and low-resource machines.

This project demonstrates how to build a complete, local RAG pipeline using:

ChromaDB for persistent vector storage

MiniLM embeddings for semantic retrieval

FLAN-T5 (small) for grounded answer generation

The system answers questions strictly from a local knowledge base, avoiding hallucinations.

🎯 Why This Project Exists

Most RAG tutorials assume:

Paid APIs

High-end GPUs

Cloud infrastructure

This project proves that:

You can build a real RAG system locally, for free, on a CPU with limited RAM, while still following clean architecture and production-style design.

🚀 Features

Semantic retrieval using all-MiniLM-L6-v2

Persistent ChromaDB vector store (no re-embedding on restart)

CPU-friendly text generation using google/flan-t5-small

Overlapping text chunking for better retrieval quality

Strict context-grounded answering (no hallucinations)

Clean, modular, interview-ready codebase

Interactive CLI-based Q&A loop

🗂 Project Structure
mini_rag/
│
├── data/
│   └── knowledge.txt          # Text knowledge base
│
├── src/
│   ├── loader/
│   │   └── loader.py          # Text loading & chunking
│   │
│   ├── db/
│   │   └── vector_store.py    # ChromaDB setup, add, query
│   │
│   ├── retrieval/
│   │   └── retriever.py       # Retrieval logic
│   │
│   ├── generation/
│   │   └── generator.py       # Prompting & answer generation
│   │
│   └── main.py                # CLI entry point
│
├── chroma_db/                 # Persistent vector store (gitignored)
├── requirements.txt
└── README.md

🔄 RAG Workflow (High-Level)

Load text from knowledge.txt

Split text into overlapping chunks

Generate embeddings using MiniLM

Store embeddings in ChromaDB

Accept user query

Embed query

Retrieve top-K relevant chunks

Generate answer using retrieved context only

🧠 Architecture Overview
knowledge.txt
      ↓
Text Loader
      ↓
Chunking (with overlap)
      ↓
Embedding Model (MiniLM)
      ↓
Vector Store (ChromaDB – persistent)
      ↓
User Query
      ↓
Query Embedding
      ↓
Similarity Search
      ↓
Top-K Relevant Chunks
      ↓
LLM (FLAN-T5)
      ↓
Final Answer

🧩 Modular Design (File-Level Responsibility)
Module	Responsibility
loader	Load raw text and split into chunks
vector_store	Manage vector DB creation, insertion, retrieval
retriever	Query orchestration & retrieval logic
generator	Context-grounded answer generation
main	Application entry point & CLI loop

Each component has a single responsibility, making the system easy to extend or replace.

⚙️ Tech Stack
Component	Technology
Embeddings	SentenceTransformers (MiniLM)
Vector DB	ChromaDB
LLM	google/flan-t5-small
Language	Python
Hardware	CPU-only
🧪 Limitations

Designed for small to medium text corpora

FLAN-T5-small has limited reasoning depth

No conversation memory (single-turn Q&A)

No re-ranking or hybrid search (yet)

🔮 Future Improvements

Conversation memory

Similarity thresholding

Re-ranking (cross-encoder)

Hybrid retrieval (BM25 + embeddings)

Evaluation metrics for retrieval quality

Web UI / API layer

📌 Key Takeaway

This project demonstrates that RAG is an architecture, not a paid API feature.
It focuses on clarity, correctness, and constraints, making it ideal for learning, interviews, and small-scale applications.