import numpy as np
import os
import pickle
import re
from datetime import datetime, timedelta
import dateutil.parser

import faiss
from datetime import datetime
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

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

# Choix du modèle pour la transformation en représentation numérique (compatible avec RAGAS)
embeddings_model = MistralAIEmbeddings(api_key = api_key, model = model_embed)

# Modèle pour générer la réponse (LLM)
chatbot_llm = ChatMistralAI(model = model_llm, api_key=api_key)

# Fonction pour l'embedding du prompt de l'utilisateur
def embed_query(query): # récupère le prompt, transforme en vecteur nupérique (token > passage dans le transformer spécialisé > extraction > normalisation > renvoi du vecteur) 
    return embeddings_model.embed_query(query) # retourne une liste de floats

# Vérifier si l'évènement est actif
def est_actif(event):
    if event["t"] is None:
        return False

    date_fin = datetime.fromisoformat(event["lastdate_end"].replace("Z", "+00:00"))
    return date_fin >= datetime.now(date_fin.tzinfo)

# Normaliser
def normalize_city(value):
    if isinstance(value, str):
        return value.strip().lower().split("(")[0].strip()
    return None

# for e in metadata:
#     if normalize_city(e.get("city")) == "strasbourg":
#         print(e["title"], e["firstdate_begin"], e["lastdate_end"], e.get("timing_begin"), e.get("timing_end"))


# Détecter automatiquement les villes
def extraire_ville(question: str):
    # Normaliser la question
    question_lower = str(question).lower()

    # Extraire toutes les villes valides (uniquement les strings)
    villes_disponibles = {
        normalize_city(e.get("city"))
        for e in metadata
        if normalize_city(e.get("city")) is not None}

    # Chercher une ville dans la question
    for ville in villes_disponibles:
        if ville in question_lower:
            return ville

    return None


# Compréhension temporelle
def extraire_intervalle_temporel(question: str):
    question = question.lower().strip()
    now = datetime.now()

    # aujourd'hui / demain / après-demain / hier
    if "aujourd'hui" in question:
        return now, now
    if "demain" in question:
        d = now + timedelta(days=1)
        return d, d
    if "après-demain" in question:
        d = now + timedelta(days=2)
        return d, d
    if "hier" in question:
        d = now - timedelta(days=1)
        return d, d

    # dans X jours / semaines / mois
    m = re.search(r"dans (\d+) jours", question)
    if m:
        n = int(m.group(1))
        d = now + timedelta(days=n)
        return d, d

    m = re.search(r"dans (\d+) semaines", question)
    if m:
        n = int(m.group(1))
        d = now + timedelta(weeks=n)
        return d, d

    m = re.search(r"dans (\d+) mois", question)
    if m:
        n = int(m.group(1))
        d = now + timedelta(days=30*n)
        return d, d

    # cette semaine / semaine prochaine
    if "cette semaine" in question:
        start = now - timedelta(days=now.weekday())
        end = start + timedelta(days=6)
        return start, end

    if "la semaine prochaine" in question:
        start = now - timedelta(days=now.weekday()) + timedelta(days=7)
        end = start + timedelta(days=6)
        return start, end

    # ce week-end
    if "week-end" in question or "weekend" in question:
        saturday = now + timedelta(days=(5 - now.weekday()) % 7)
        sunday = saturday + timedelta(days=1)
        return saturday, sunday

    # mois prochain
    if "mois prochain" in question:
        start = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
        end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        return start, end

    # en septembre 2026
    m = re.search(r"en ([a-zéû]+) (\d{4})", question)
    if m:
        mois_nom = m.group(1)
        annee = int(m.group(2))
        try:
            mois_num = dateutil.parser.parse(mois_nom).month
            start = datetime(annee, mois_num, 1)
            end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            return start, end
        except:
            pass

    # le 12 août 2026
    try:
        d = dateutil.parser.parse(question, fuzzy=True)
        return d, d
    except:
        pass

    # du 5 au 7 juillet
    m = re.search(r"du (\d{1,2}) au (\d{1,2}) ([a-zéû]+)", question)
    if m:
        jour1 = int(m.group(1))
        jour2 = int(m.group(2))
        mois_nom = m.group(3)
        try:
            mois_num = dateutil.parser.parse(mois_nom).month
            start = datetime(now.year, mois_num, jour1)
            end = datetime(now.year, mois_num, jour2)
            return start, end
        except:
            pass
    return None

# Filtre temporel
def event_is_in_interval(event, start, end):
    try:
        debut = datetime.fromisoformat(event["timing_begin"].replace("Z", "+00:00"))
        fin = datetime.fromisoformat(event["timing_end"].replace("Z", "+00:00"))
        return (debut <= end) and (fin >= start)
    except:
        return False


# Recherche des évènements pertinents en fonction de l'actualité de l'évènement au moment de l'envoi du prompt
def recherche_event_pertinent(query, k = 40, max_results = 5):
    q_emb = embed_query(query) # transforme le prompt en numérique
    distances, indices = index.search(np.array(q_emb, dtype="float32").reshape(1, -1), k) # transforme la liste de floats en tableau numpy avec vecteurs et dimension

    intervalle = extraire_intervalle_temporel(query)
    ville = extraire_ville(query)

    # extraction âge
    m = re.search(r"(\d+)\s*ans", query.lower())
    age_min = int(m.group(1)) if m else None

    # extraction mots-clés
    mots_cles = re.findall(r"[a-zA-Zéèêàùûôç]+", query.lower())

    results = []
    uid_vu = set()

    for idx in indices[0]:
        event = metadata[idx]

        # éviter doublons
        if event["uid"] in uid_vu:
            continue

        # filtre actif
        try:
            date_fin = datetime.fromisoformat(event["lastdate_end"].replace("Z", "+00:00"))
            if date_fin < datetime.now(date_fin.tzinfo):
                continue
        except:
            continue

        # filtre ville
        city_event = normalize_city(event.get("city"))
        if ville is not None:
            if city_event != ville:
                continue

        # # filtre âge
        if age_min is not None:
            if event.get("age_minimum", 0) > age_min:
                continue

        # filtre temporel
        if intervalle is not None:
            start, end = intervalle
            if not event_is_in_interval(event, start, end):
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
Tu ne connais AUCUN événement en dehors du contexte fourni.

Voici la question de l'utilisateur:
{question}

Voici des informations issues de la base d'événements (contexte RAG):
{context}

Consignes:
- Tu ne dois utiliser QUE les événements présents dans le contexte.
- Il est interdit d'utiliser tes connaissances générales.
- Il est interdit d'inventer un événement.
- Tu n'as pas accès à Internet.
- Si le contexte est vide, réponds exactement: "Je n'ai trouvé aucun événement correspondant à votre recherche dans la base de données."
- Mentionne le titre, la ville, les dates, les conditions, l'âge minimum et maximum et le lien quand c'est possible.

Réponse :
""")

# Construire le contexte
def build_context(results): # contruit un contexte complet qui fournit toutes les informations nécessaires
    parts = []
    for r in results:
        part = (
            f"Titre: {r['title']}\n"
            f"Ville: {r['city']}\n"
            f"Lieu: {r.get('lieu', '')}\n"
            # f"Adresse: {r.get('location_address', '')}\n"
            f"Conditions: {r.get('conditions', '')}\n"
            f"Age minimum: {r.get('age_minimum', '')}\n"
            f"Age maximum: {r.get('age_maximum', '')}\n"
            f"Debut: {r['timing_begin']}\n"
            f"Fin: {r['timing_end']}\n"
            f"Dates: {r['date']}\n"
            f"Lien: {r['canonicalurl']}\n"
            f"Description (extrait):\n{r['chunk']}\n"
            "-----------------------------\n")
        parts.append(part)
    return "\n".join(parts)

# RAG
def generate_answer(question):
    event = recherche_event_pertinent(question, max_results = 10)
    if len(event) == 0:
        return "Je n'ai trouvé aucun événement correspondant à votre recherche."
    
    context = build_context(event)

    chain = prompt | chatbot_llm
    response = chain.invoke(
        {
            "context": context,
            "question": question
        })

    return response.content

# Classe pour API
class PulsEventRAG:
    def __init__(self):
        # On réutilise directement TON index FAISS et TES metadata
        self.index = index
        self.metadata = metadata

        # Embeddings déjà configurés
        self.embeddings = embeddings_model

        # LLM déjà configuré
        self.llm = chatbot_llm

    def rebuild_index(self):
        # Recharge FAISS et metadata depuis les fichiers
        self.index = faiss.read_index("./faiss_index/faiss.idx")
        with open("./faiss_index/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        return "Index reconstruit avec succès."

    def ask(self, question: str) -> str:
        return generate_answer(question)

# Tests
# print(generate_answer("Je cherche un atelier pour un enfant à Reims."))
print(generate_answer("Quels événements sont prévus le mois prochain à Strasbourg?"))
# print(generate_answer("Y a-t-il des concerts gratuits à Metz?"))
# print(generate_answer("Quels événements sont adaptés aux seniors à Nancy?"))
# print(generate_answer("Que faire en famille à Mulhouse en septembre?"))
# print(generate_answer("Je cherche des ateliers créatifs à Champagne et Charleville-Mézières."))
