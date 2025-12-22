# Imported libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1️ Load and chunk the knowledge
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=50):
    """
    Split text into small chunks (50 words per chunk recommended for CPU)
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# 2️ Initialize lightweight embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Small, fast, CPU-friendly

def embed_chunks(chunks):
    return model.encode(chunks)

# 3️ Retrieve relevant chunks
def retrieve_chunks(query, chunks, embeddings, top_k=1):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # highest first
    return [chunks[i] for i in top_indices]

# 4️ Main loop: Q&A
if __name__ == "__main__":
    print("Loading knowledge base...")
    text = load_text("data/knowledge.txt")
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    print(f"Knowledge base loaded with {len(chunks)} chunks.")

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        relevant_chunks = retrieve_chunks(query, chunks, embeddings)
        print("\nAnswer (from knowledge):")
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"{i}. {chunk}")