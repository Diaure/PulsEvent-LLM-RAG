import pandas as pd
import pickle
import numpy as np
import faiss

# Chargement des chnks
with open("./data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Génération des embeddings pour chaque document
embeddings = []

for e in chunks:
    embeddings.append(e["chunk"])   # <-- boucle simple comme tu voulais

vectors = np.array(embeddings).astype("float32")
print("Taille des vecteurs :", embeddings.shape)

# Initialisation de l'index Faiss
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)

# Ajout des embeddings à l'index
index.add(vectors)

# Sauvegarde de l'index sur le disque
faiss.write_index(index, "./faiss_index/faiss.idx")

# Sauvegarde des métadonnées
metadata = [
    {"uid": e["uid"],
        "title": e["title"],
        "city": e["city"],
        "date": e["date"],
        "canonicalurl": e["canonicalurl"],
        "chunk": e["chunk"]}
    for e in embeddings]

with open("./faiss_index/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
