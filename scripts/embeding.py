import os
import pickle
from mistralai import Mistral
import tqdm
from dotenv import load_dotenv
import time

# Chargement des clés
load_dotenv()

# Charger les chunks
with open("./data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Nombre de chunks à vectoriser : {len(chunks)}")

# Initialiser Mistral
api_key = os.getenv("PULSEVENT_MISTRAL_KEY")
if not api_key:
    raise ValueError("❌ La clé API n'est pas chargée depuis .env")
client = Mistral(api_key=api_key)
model = "mistral-embed"

# Envoi des requêtes par batch
BATCH_SIZE = 64
embeddings = []
SLEEP_TIME = 1.0 

def embed_batch(batch):
    texts = [c["chunk"] for c in batch]
    while True:
        try:
            response = client.embeddings.create(
                model=model,
                inputs=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            if "429" in str(e):
                print("Rate limit atteint → pause 5 secondes…")
                time.sleep(5)
            else:
                raise e

# Vectorisation par batch
for i in tqdm.tqdm(range(0, len(chunks), BATCH_SIZE), desc="Vectorisation Mistral"):
    batch = chunks[i:i+BATCH_SIZE]
    batch_embeddings = embed_batch(batch)
    
    for c, emb in zip(batch, batch_embeddings):
        embeddings.append({
            "uid": c["uid"],
            "title": c["title"],
            "city": c["city"],
            "lieu": c["lieu"],
            "date": c["date"],
            "lastdate_end": c["lastdate_end"],
            "conditions": c["conditions"],
            "age_minimum": c["age_minimum"],
            "age_maximum": c["age_maximum"],
            "canonicalurl": c["canonicalurl"],
            "chunk": c["chunk"],
            "embedding": emb})
    
    time.sleep(SLEEP_TIME)

# Sauvegarde
with open("./data/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
