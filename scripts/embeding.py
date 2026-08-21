import os
import pickle
from mistralai.client import Mistral
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
BATCH_SIZE = 32
embeddings = []
SLEEP_TIME = 1.0 
CHECKPOINT_FILE = "./data/embeddings_checkpoint.pkl"

# Charger un éventuel checkpoint
if os.path.exists(CHECKPOINT_FILE):

    with open(CHECKPOINT_FILE, "rb") as f:
        embeddings = pickle.load(f)

    print(
        f"Checkpoint trouvé : "
        f"{len(embeddings)} embeddings déjà calculés"
    )

else:
    embeddings = []

# Nombre de chunks déjà traités
start = len(embeddings)

print(f"Reprise à partir du chunk : {start}")

def embed_batch(batch):
    texts = [c["chunk"] for c in batch]

    while True:
        try:
            response = client.embeddings.create(
                model=model,
                inputs=texts
            )

            return [item.embedding for item in response.data]

        except Exception as e:
            error = str(e)

            if "429" in error:
                print("Rate limit atteint → pause 10 secondes...")
                time.sleep(10)

            elif "ReadTimeout" in error or "timed out" in error.lower():
                print("Timeout Mistral → nouvelle tentative dans 10 secondes...")
                time.sleep(10)

            else:
                raise

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
            "timing_begin": c["timing_begin"],
            "timing_end": c["timing_end"],
            "firstdate_begin": c["firstdate_begin"],
            "lastdate_end": c["lastdate_end"],
            "conditions": c["conditions"],
            "age_minimum": c["age_minimum"],
            "age_maximum": c["age_maximum"],
            "canonicalurl": c["canonicalurl"],
            "keywords_fr": c["keywords_fr"],
            "description_fr": c["description_fr"],
            "longdescription_fr": c["longdescription_fr"],
            "chunk": c["chunk"],
            "embedding": emb})
    
    time.sleep(SLEEP_TIME)

# Sauvegarde
with open("./data/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
