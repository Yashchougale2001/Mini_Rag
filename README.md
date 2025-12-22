# Mini RAG Project

A simple Retrieval Augmented Generation (RAG) system using:
- Sentence Transformers
- Cosine similarity
- OpenAI LLM

Answers questions strictly from a local text file.

1. Load text
2. Split into chunks
3. Embed chunks
4. Store embeddings
5. Accept user query
6. Embed query
7. Retrieve relevant chunks
8. Ask LLM using retrieved text

Workflow

knowledge.txt
      ↓
Text Loader
      ↓
Text Chunking
      ↓
Embedding Model (MiniLM)
      ↓
Vector Store (in-memory)
      ↓
User Query
      ↓
Query Embedding
      ↓
Similarity Search (cosine)
      ↓
Top-K Relevant Chunks
      ↓
Displayed Answer (retrieved text)
