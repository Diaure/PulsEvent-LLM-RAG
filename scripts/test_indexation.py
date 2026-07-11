import pickle
import numpy as np
import faiss
from mistralai import Mistral
import os
from dotenv import load_dotenv

# Chargement des clés
load_dotenv()

# Charger l’index FAISS
index = faiss.read_index("./faiss_index/faiss.idx")

# Charger les métadonnées
with open("./faiss_index/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"Index chargé : {index.ntotal} vecteurs")
print(f"Métadonnées chargées : {len(metadata)} événements")

# Initialiser Mistral pour embed la requête
api_key = os.getenv("PULSEVENT_MISTRAL_KEY")
if not api_key:
    raise ValueError("❌ La clé API n'est pas chargée depuis .env")
client = Mistral(api_key=api_key)
model = "mistral-embed"

def embed_query(query: str):
    response = client.embeddings.create(
        model=model,
        inputs=[query]
    )
    return np.array(response.data[0].embedding).astype("float32")

def search(query: str, k=5):
    print(f"\n🔎 Recherche : {query}")

    # Embedding de la requête
    q_emb = embed_query(query).reshape(1, -1)

    # Recherche FAISS
    distances, indices = index.search(q_emb, k)

    print("\nRésultats :")
    for rank, idx in enumerate(indices[0]):
        event = metadata[idx]
        print(f"\n--- Résultat {rank+1} ---")
        print(f"Titre : {event['title']}")
        print(f"Ville : {event['city']}")
        print(f"Date : {event['date']}")
        print(f"URL : {event['canonicalurl']}")
        print(f"Chunk : {event['chunk'][:200]}...")
        print(f"Distance : {distances[0][rank]:.4f}")

# Tests adaptés au Grand Est
if __name__ == "__main__":
    search("concert à Strasbourg")
    search("atelier pour enfants à Reims")
    search("événement gastronomique Alsace")
    search("festival de musique Metz")
    search("exposition art contemporain Mulhouse")
