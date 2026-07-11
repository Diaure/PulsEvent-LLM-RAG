import os
import pickle
from mistralai import Mistral
import tqdm

# Charger les chunks
with open("./data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Nombre de chunks à vectoriser : {len(chunks)}")

# Initialiser Mistral
api_key = os.getenv("PULSEVENT_MISTRAL_KEY")
client = Mistral(api_key=api_key)
model = "mistral-embed"

def embed_text(text):
    response = client.embeddings.create(
        model=model,
        inputs=[text]
    )
    return response.data[0].embedding

# Vectorisation
embeddings = []
for c in tqdm.tqdm(chunks, desc="Vectorisation Mistral"):
    emb = embed_text(c["chunk"])
    embeddings.append({
        "uid": c["uid"],
        "title": c["title"],
        "city": c["city"],
        "date": c["date"],
        "canonicalurl": c["canonicalurl"],
        "chunk": c["chunk"],
        "embedding": emb})

# Sauvegarde
with open("./data/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
print("Embeddings sauvegardés dans ./data/embeddings.pkl")
