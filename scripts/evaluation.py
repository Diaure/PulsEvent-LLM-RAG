import numpy as np
from mistralai import Mistral
from dotenv import load_dotenv
import os

from chatbot_rag import generate_answer  # ton chatbot

load_dotenv()
api_key = os.getenv("PULSEVENT_MISTRAL_KEY")

embed_client = Mistral(api_key=api_key)
model_embed = "mistral-embed"

# Fonction pour embedder un texte
def embed_text(text):
    response = embed_client.embeddings.create(
        model=model_embed,
        inputs=[text]
    )
    return np.array(response.data[0].embedding, dtype="float32")

# Similarité cosinus
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Classification automatique
def classify(similarity):
    if similarity >= 0.85:
        return "correcte"
    elif similarity >= 0.65:
        return "partiellement correcte"
    else:
        return "incorrecte"

# Jeu de tests
tests = [
    {
        "question": "Je cherche un atelier pour un enfant à Reims.",
        "expected": "Réponse humaine annotée ici..."
    },
    {
        "question": "Quels événements sont prévus ce week-end à Strasbourg ?",
        "expected": "Réponse humaine annotée ici..."
    }
]

# Évaluation
for test in tests:
    print("\n==============================")
    print("Question :", test["question"])

    ia_answer = generate_answer(test["question"])
    expected = test["expected"]

    print("\nRéponse IA :\n", ia_answer)
    print("\nRéponse humaine :\n", expected)

    emb_ia = embed_text(ia_answer)
    emb_expected = embed_text(expected)

    sim = cosine_similarity(emb_ia, emb_expected)
    label = classify(sim)

    print("\nSimilarité :", round(sim, 3))
    print("Évaluation :", label)
