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
