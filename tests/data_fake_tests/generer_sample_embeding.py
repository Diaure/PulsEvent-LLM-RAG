import numpy as np
import pickle

fake_embeddings = [
    {"uid": "1", "embedding": np.random.rand(384).tolist()},
    {"uid": "2", "embedding": np.random.rand(384).tolist()}
]

with open("fake_embeddings.pkl", "wb") as f:
    pickle.dump(fake_embeddings, f)

print("fake_embeddings.pkl créé.")
