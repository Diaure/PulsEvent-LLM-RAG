import numpy as np
import os
import pickle
import re
from datetime import datetime, timedelta, timezone
import dateutil.parser
import unicodedata

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
def load_faiss_index():
    # Charge FAISS en local, mock en CI/CD.
    if os.getenv("CI") == "true":
        # Mock FAISS index pour CI/CD
        d = 384
        index = faiss.IndexFlatL2(d)

        metadata = []
        return index, metadata
    else:
        index = faiss.read_index("./faiss_index/faiss.idx")
        with open("./faiss_index/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            # print(metadata[0]['city'])
        return index, metadata


# Configuration de la connexion à l'API Mistral
embed_client = Mistral(api_key=api_key)
model_embed = "mistral-embed"
model_llm = "mistral-large-latest"

if os.getenv("CI") == "true":
    class MockEmbeddings:
        def embed_query(self, text):
            return np.random.rand(384).tolist()
    embeddings_model = MockEmbeddings()

    class MockLLM:
        def invoke(self, text):
            return "Réponse mock CI/CD"
    chatbot_llm = MockLLM()

else:
    # Choix du modèle pour la transformation en représentation numérique (compatible avec RAGAS)
    embeddings_model = MistralAIEmbeddings(api_key = api_key, model = model_embed)

    # Modèle pour générer la réponse (LLM)
    chatbot_llm = ChatMistralAI(model = model_llm, api_key=api_key)

# Fonction pour l'embedding du prompt de l'utilisateur
def embed_query(query): # récupère le prompt, transforme en vecteur nupérique (token > passage dans le transformer spécialisé > extraction > normalisation > renvoi du vecteur) 
    return embeddings_model.embed_query(query) # retourne une liste de floats

# Vérifier si l'évènement est actif
def est_actif(event):
    if event["title"] is None:
        return False

    date_fin = datetime.fromisoformat(event["lastdate_end"].replace("Z", "+00:00"))
    return date_fin >= datetime.now(date_fin.tzinfo)

# Normaliser
def normalize_city(value):
    if isinstance(value, str):
        return value.strip().lower().split("(")[0].strip()
    return None


# extraction automatique des villes
def extraire_ville(question):
    index, metadata = load_faiss_index()
    # Normaliser la question
    q = unicodedata.normalize("NFD", question.lower())
    q = ''.join(c for c in q if unicodedata.category(c) != "Mn")

    # Extraire toutes les villes valides
    villes_disponibles = list(set(
        normalize_city(e.get("city"))
        for e in metadata
        if normalize_city(e.get("city")) is not None))

    # Chercher une ville dans la question
    villes_disponibles.sort(key=len, reverse=True)
    for ville in villes_disponibles:
        pattern = r"\b" + re.escape(ville) + r"\b"
        if re.search(pattern, q):
            return ville
    return None


# Compréhension temporelle
def extraire_intervalle_temporel(question: str):
    question = question.lower().strip()
    now = datetime.now(timezone.utc)

    def aware(dt):
        # Convertit n'importe quelle date naive en UTC aware
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

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
        start = aware(now - timedelta(days=now.weekday()))
        end = aware(start + timedelta(days=6))
        return start, end

    if "la semaine prochaine" in question:
        start = aware(now - timedelta(days=now.weekday()) + timedelta(days=7))
        end = aware(start + timedelta(days=6))
        return start, end

    # ce week-end
    if "week-end" in question or "weekend" in question:
        saturday = aware(now + timedelta(days=(5 - now.weekday()) % 7))
        sunday = aware(saturday + timedelta(days=1))
        return saturday, sunday

    # mois prochain
    if "mois prochain" in question:
        start = aware((now.replace(day=1) + timedelta(days=32)).replace(day=1))
        end = aware((start + timedelta(days=32)).replace(day=1) - timedelta(days=1))
        return start, end

    # en septembre 2026
    m = re.search(r"en ([a-zéû]+) (\d{4})", question)
    if m:
        mois_nom = m.group(1)
        annee = int(m.group(2))
        try:
            mois_num = dateutil.parser.parse(mois_nom).month
            start = aware(datetime(annee, mois_num, 1))
            end = aware((start + timedelta(days=32)).replace(day=1) - timedelta(days=1))
            return start, end
        except:
            pass

    # le 12 août 2026
    # try:
    #     d = dateutil.parser.parse(question, fuzzy=True)
    #     return aware(d), aware(d)
    # except:
    #     pass

    # du 5 au 7 juillet
    m = re.search(r"du (\d{1,2}) au (\d{1,2}) ([a-zéû]+)", question)
    if m:
        jour1 = int(m.group(1))
        jour2 = int(m.group(2))
        mois_nom = m.group(3)
        try:
            mois_num = dateutil.parser.parse(mois_nom).month
            start = aware(datetime(now.year, mois_num, jour1))
            end = aware(datetime(now.year, mois_num, jour2))
            return start, end
        except:
            pass

    return None


# Filtre temporel
def event_is_in_interval(event, start, end):
    print("\n----------------")
    print(event["title"])
    try:
        # Essayer timing_begin / timing_end
        debut_raw = event.get("timing_begin")
        fin_raw = event.get("timing_end")

        if debut_raw and fin_raw:
            debut = datetime.fromisoformat(debut_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
            fin = datetime.fromisoformat(fin_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        else:
            # Fallback: firstdate_begin / lastdate_end
            debut = datetime.fromisoformat(event["firstdate_begin"].replace("Z", "+00:00")).astimezone(timezone.utc)
            fin = datetime.fromisoformat(event["lastdate_end"].replace("Z", "+00:00")).astimezone(timezone.utc)

        print("Début event :", debut)
        print("Fin event :", fin)
        print("Start :", start)
        print("End :", end)
        
        print((debut <= end) and (fin >= start))

        return (debut <= end) and (fin >= start)

    except Exception as e:
        print("ERREUR :", e)
        return False


# Recherche des évènements pertinents en fonction de l'actualité de l'évènement au moment de l'envoi du prompt
def recherche_event_pertinent(query, k = 100, max_results = 20):
    index, metadata = load_faiss_index()
    q_emb = embed_query(query) # transforme le prompt en numérique
    distances, indices = index.search(np.array(q_emb, dtype="float32").reshape(1, -1), k) # transforme la liste de floats en tableau numpy avec vecteurs et dimension

    intervalle = extraire_intervalle_temporel(query)
    ville = extraire_ville(query)

    print("VILLE EXTRAITE :", ville)
    print("INTERVALLE :", intervalle)

    # extraction âge
    m = re.search(r"(\d+)\s*ans", query.lower())
    age_demande = int(m.group(1)) if m else None

    # extraction mots-clés
    # mots_cles = re.findall(r"[a-zA-Zéèêàùûôç]+", query.lower())

    results = []
    uid_vu = set()

    nb_actif = 0
    nb_ville = 0
    nb_date = 0
    nb_age = 0

    for idx in indices[0]:
        event = metadata[idx]
        
        print("\n---------------------")
        print("\nEvent:", event["title"])

        # éviter doublons
        if event["uid"] in uid_vu:
            print("REJET uid")
            continue

        # filtre actif
        try:
            date_fin = datetime.fromisoformat(event["lastdate_end"].replace("Z", "+00:00"))
            if date_fin < datetime.now(date_fin.tzinfo):
                continue
        except:
            continue

        # filtre actif
        if not est_actif(event):
            print("-> rejet actif")
            continue
        print("-> actif OK")
        nb_actif += 1

        # filtre ville
        print("Ville question :", ville)
        print("Ville event :", event["city"])
        if ville is not None:
            if normalize_city(event.get("city")) != ville:
                print("-> rejet ville")
                continue
        nb_ville += 1
        print("-> ville OK")

        # # filtre âge
        if age_demande is not None:
            age_min_event = event.get("age_minimum")
            age_max_event = event.get("age_maximum")

            # si l'événement ne possède pas d'information d'âge
            if age_min_event is None and age_max_event is None:
                continue

            # si seul l'âge minimum est renseigné
            if age_min_event is not None and age_demande < age_min_event:
                continue

            # si seul l'âge maximum est renseigné
            if age_max_event is not None and age_demande > age_max_event:
                        continue
        nb_age += 1
        print("-> âge OK")

        # filtre temporel
        if intervalle is not None:
            print("-> passage filtre temporel")
            start, end = intervalle
            if not event_is_in_interval(event, start, end):
                print("-> rejet temporel")
                continue
        nb_date += 1
        print("-> temporel OK")
           
        uid_vu.add(event["uid"])
        results.append(event)

        if len(results) == max_results:
            break

    print("\n========== DEBUG ==========")
    print("Ville détectée :", ville)
    print("Intervalle :", intervalle)
    print("Après filtre actif :", nb_actif)
    print("Après filtre ville :", nb_ville)
    print("Après filtre âge :", nb_age)
    print("Après filtre temporel :", nb_date)
    print("Résultats finaux :", len(results))
    print("===========================\n")
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
        self.index, self.metadata = load_faiss_index()

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
# print(generate_answer("Je cherche un atelier pour un enfant à Reims le mois prochaine."))
# print(generate_answer("Quels événements sont prévus le mois prochain à Strasbourg ?"))
# print(generate_answer("Y a-t-il des concerts gratuits à Metz?"))
# print(generate_answer("Quels événements sont adaptés aux seniors à Nancy?"))
# print(generate_answer("Que faire en famille à Mulhouse en septembre?"))
print(generate_answer("Je cherche un cours de plongée sous-marine à Reims."))
