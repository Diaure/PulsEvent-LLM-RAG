import numpy as np
import os
import pickle
import faiss
from datetime import datetime
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

# Chargement clé API
load_dotenv()
api_key = os.getenv("PULSEVENT_MISTRAL_KEY")

# Chargement des index + metadonnées
index = faiss.read_index("./faiss_index/faiss.idx")

with open("./faiss_index/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(index.ntotal)
print(len(metadata))

# Configuration de la connexion à l'API Mistral
embed_client = Mistral(api_key=api_key)
model_embed = "mistral-embed"
model_llm = "mistral-large-latest"

# Choix du modèle pour la transformation en représentation numérique
# embeddings_model = MistralAIEmbeddings(api_key = api_key, model = model_embed)

# Modèle pour générer la réponse (LLM)
chatbot_llm = ChatMistralAI(model = model_llm, api_key=api_key)

# Fonction pour l'embedding du prompt de l'utilisateur
def embed_query(query):
    response = embed_client.embeddings.create(model = model_embed, inputs=[query])
    return response.data[0].embedding # les objets sont des listes de floats

# Vérifier si l'évènement est actif
def est_actif(event):
    if event["lastdate_end"] is None:
        return False

    date_fin = datetime.fromisoformat(event["lastdate_end"].replace("Z", "+00:00"))
    return date_fin >= datetime.now(date_fin.tzinfo)

# Recherche des évènements pertinents en fonction de l'actualité de l'évènement au moment de l'envoi du prompt
def recherche_event_pertinent(query, k = 20, max_results = 5):
    q_emb = embed_query(query) # transforme le prompt en numérique
    distances, indices = index.search(np.array(q_emb, dtype="float32").reshape(1, -1), k) # transforme la liste de floats en tableau numpy avec vecteurs et dimension

    results = []
    uid_vu = set()
    for rank, idx in enumerate(indices[0]): # enumerate renvoie la position(rank) dans la liste et la valeur(idx)
        event = metadata[idx]
        if event["uid"] in uid_vu:
            continue

        if not est_actif(event):
            continue

        uid_vu.add(event["uid"])
        results.append(event)

        if len(results) == max_results:
            break

    return results

# Définition du prompt
prompt = ChatPromptTemplate.from_template(
"""
Tu es un assistant qui recommande des événements culturels dans la région Grand Est (France).
Tu dois répondre en français, de manière claire et utile.

Voici la question de l'utilisateur:
{question}

Voici des informations issues de la base d'événements (contexte RAG):
{context}

Consignes:
- Propose les événements les plus pertinents pour la question.
- Mentionne le titre, la ville, les dates, les conditions, l'âge minimum et maximum et le lien quand c'est possible.
- Si les événements ne correspondent pas exactement, explique-le et propose les plus proches.
- Si un événement ne correspond pas aux contraintes (âge, ville, date), explique pourquoi.
- Si aucun événement n'est pertinent, indique-le aussi clairement.
- Ne fabrique pas d'événements qui ne sont pas dans le contexte.
Réponse :
""")

# Construire le contexte
def build_context(results): # contruit un contexte complet qui fournit toutes les informations nécessaires
    parts = []
    for r in results:
        part = (
            f"Titre: {r['title']}\n"
            f"Ville: {r['city']}\n"
            f"Lieu: {r.get('location_name', '')}\n"
            f"Adresse: {r.get('location_address', '')}\n"
            f"Conditions: {r.get('conditions_fr', '')}\n"
            f"Age minimum: {r.get('age_min', '')}\n"
            f"Age maximum: {r.get('age_max', '')}\n"
            f"Dates: {r['date']}\n"
            f"Lien: {r['canonicalurl']}\n"
            f"Description (extrait):\n{r['chunk']}\n"
            "-----------------------------\n")
        parts.append(part)
    return "\n".join(parts)

# RAG
def generate_answer(question):
    event = recherche_event_pertinent(question, max_results = 10)
    context = build_context(event)

    chain = prompt | chatbot_llm
    response = chain.invoke(
        {
            "context": context,
            "question": question
        })

    return response.content

# Tests
# print(generate_answer("Je cherche un concert à Strasbourg."))
print(generate_answer("Je cherche un atelier pour un enfant à Reims."))
# print(generate_answer("Quels événements sont prévus ce week-end à Strasbourg ?"))