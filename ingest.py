from sentence_transformers import SentenceTransformer
import os
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

DB_PATH = "endee_db.json"

def load_notes(folder_path="data"):
    texts = []
    metadata = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                content = f.read()
                chunks = content.split("\n\n")

                for chunk in chunks:
                    if chunk.strip():
                        texts.append(chunk)
                        metadata.append({"source": file_name})

    return texts, metadata

texts, metadata = load_notes()

embeddings = model.encode(texts).tolist()

records = []
for i, text in enumerate(texts):
    records.append({
        "id": i,
        "text": text,
        "embedding": embeddings[i],
        "metadata": metadata[i]
    })

with open(DB_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print(" Notes stored successfully in Endee-style Vector Database!")
