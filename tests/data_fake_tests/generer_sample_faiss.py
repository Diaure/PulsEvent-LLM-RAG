import pickle
import numpy as np
import faiss

with open("tests/fake_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

vectors = np.array([d["embedding"] for d in data]).astype("float32")

dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

faiss.write_index(index, "fake_faiss.idx")

print("fake_faiss.idx créé.")
