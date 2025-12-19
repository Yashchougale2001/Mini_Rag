from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai

#Load and chunk text
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks


#Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    return model.encode(chunks)


#Retrieve relevant chunks
def retrieve_chunks(query, chunks, embeddings, top_k=2):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:]
    return [chunks[i] for i in top_indices]


#Ask LLM using retrieved context
openai.api_key = "YOUR_API_KEY"

def ask_llm(context, question):
    prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response["choices"][0]["message"]["content"]


#Main execution
if __name__ == "__main__":
    text = load_text("knowledge.txt")
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        relevant_chunks = retrieve_chunks(query, chunks, embeddings)
        context = "\n".join(relevant_chunks)

        answer = ask_llm(context, query)
        print("\nAnswer:", answer)
