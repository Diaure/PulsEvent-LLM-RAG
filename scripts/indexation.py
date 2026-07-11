import pandas as pd
import pickle
import numpy as np
import faiss

# Chargement des chnks
with open("./data/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Nombre d'embeddings chargés : {len(data)}")

# Extraire les vecteurs
vectors = np.array([d["embedding"] for d in data]).astype("float32")
print("Taille des vecteurs :", vectors.shape)

# Initialisation de l'index Faiss
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)

# Ajout des embeddings à l'index
index.add(vectors)

# Sauvegarde de l'index sur le disque
faiss.write_index(index, "./faiss_index/faiss.idx")
print("Index sauvegardé dans faiss_index/faiss.idx")

# Sauvegarde des métadonnées
metadata = [
    {"uid": e["uid"],
        "title": e["title"],
        "city": e["city"],
        "date": e["date"],
        "canonicalurl": e["canonicalurl"],
        "chunk": e["chunk"]}
    for e in data]

with open("./faiss_index/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
