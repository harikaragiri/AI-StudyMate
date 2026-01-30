from sentence_transformers import SentenceTransformer
import json
import numpy as np

DB_PATH = "endee_db.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load vector database
with open(DB_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

texts = [r["text"] for r in records]
embeddings = np.array([r["embedding"] for r in records])

def semantic_search(query, top_k=3):
    query_embedding = model.encode([query])[0]

    scores = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "text": texts[idx],
            "score": float(scores[idx])
        })

    return results


# Test search
if __name__ == "__main__":
    user_query = input("Ask a question: ")
    results = semantic_search(user_query)

    print("\n🔍 Top Matches:\n")
    for r in results:
        print("-", r["text"])
