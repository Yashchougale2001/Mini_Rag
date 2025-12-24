from src.loader.loader import load_text, chunk_text
from src.db.vector_store import create_vector_store, add_chunks_to_db, query_chunks
from src.retrieval.retriever import retrieve_chunks
from src.generation.generator import generate_answer

if __name__ == "__main__":
    print("Loading knowledge base...")
    text = load_text("data/knowledge.txt")
    chunks = chunk_text(text)

    vector_store = create_vector_store()
    add_chunks_to_db(vector_store, chunks)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        context = retrieve_chunks(vector_store, query, top_k=3)
        answer = generate_answer(query, context)

        print("\nAnswer:")
        print(answer)
