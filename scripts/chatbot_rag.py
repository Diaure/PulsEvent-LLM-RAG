import numpy as np
import json
import requests
import os
import pickle
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter
import dateutil.parser
import unicodedata

import faiss
from datetime import datetime
import time
from dotenv import load_dotenv
from mistralai.client import Mistral
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

        metadata = [
        {
            "uid": i,
            "title": f"Mock Event {i}",
            "city": "reims",
            "canonicalurl": "https://example.com",
            "chunk": "mock chunk",
            "lieu": "mock",
            "date": "2026-01-01",
            "timing_begin": "10:00",
            "timing_end": "18:00",
            "firstdate_begin": "2026-01-01T10:00:00Z",
            "lastdate_end": "2099-01-01T00:00:00Z",
            "conditions": "",
            "age_minimum": 0,
            "age_maximum": 99
        } for i in range(200)]

        return index, metadata
    else:
        index = faiss.read_index("./faiss_index/faiss.idx")
        with open("./faiss_index/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
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

    # Récupérer les villes de la BDD
    villes_disponibles = set()
    for event in metadata:
        ville = normalize_city(event.get("city"))

        if ville is not None:
            villes_disponibles.add(ville)

    # Chercher une ville connue dans la question
    for ville in sorted(villes_disponibles, key=len, reverse=True):
        ville_normalisee = unicodedata.normalize("NFD", ville.lower())
        ville_normalisee = ''.join(c for c in ville_normalisee if unicodedata.category(c) != "Mn")

        pattern = r"\b" + re.escape(ville_normalisee) + r"\b"
        if re.search(pattern, q):
            return ville

    match = re.search(
        r"\b(?:à|a|dans)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]*(?:[\s-][A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]*)*)",
        question,
        re.IGNORECASE)
    
    if match:
        ville_detectee = normalize_city(match.group(1).strip())
        ville_detectee = normalize_city(ville_detectee)
        return ville_detectee

    # Si aucune ville connue dans la quesion
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
    print(event.get("keywords_fr"))
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

# fonction permettant de rechercher par mots clés
# type de mots-clés
type_keywords = {
    "atelier": ["atelier", "créatif", "création", "initiation", "fabrication", "cuisine", "maquette", "bricolage"],

    "concert": ["concert", "musique", "musicale", "musical", "musicales", "musique classique", "musique ancienne",
        "chorale",  "choral", "chœur", "choeur","orchestre", "orchestre symphonique", "orchestre de chambre",
        "récital", "recital", "opéra", "opera", "opérette", "operette", "symphonie", "symphonique", "jazz", 
        "chant", "chants", "vocal", "vocale", "lyrique"],

    "exposition": ["exposition", "vernissage", "musée", "galerie"],

    "visite": ["visite", "visite guidée", "patrimoine", "découverte"],

    "spectacle": ["spectacle", "théâtre", "danse", "marionnette", "cirque"]}

def determiner_type_evenement(event):
    textes = []

    # keywords_fr
    keywords = event.get("keywords_fr")

    if keywords is not None and str(keywords).lower() != "nan":
        if isinstance(keywords, list):
            textes.extend(str(k) for k in keywords)
        else:
            textes.append(str(keywords))

    # titre
    textes.append(str(event.get("title", "")))

    # description courte
    textes.append(str(event.get("description", "")))

    # description longue
    textes.append(str(event.get("longdescription", "")))

    texte = " ".join(textes).lower()

    types_detectes = []

    for categorie, mots in type_keywords.items():
        if any(
            re.search(
                r"\b" + re.escape(mot.lower()) + r"\b",
                texte
            )
            for mot in mots
        ):
            types_detectes.append(categorie)

    return types_detectes if types_detectes else None

def extraire_type_evenement(question):
    question = question.lower()

    # Type connu dans la base
    for categorie, mots in type_keywords.items():
        for mot in mots:
            if re.search(r"\b" + re.escape(mot) + r"\b", question):
                return {
                    "type": categorie,
                    "statut": "connu"}

    # Déterminer si la question contient une demande d'activité/type spécifique
    motifs_type_demande = [r"\bcours\s+de\s+(.+)", r"\bcours\s+d['’](.+)", 
                           r"\bformation\s+(?:en|de|à|a)\s+(.+)", r"\bstage\s+(?:en|de|à|a)\s+(.+)",
                            r"\binitiation\s+(?:à|a)\s+(.+)",]
    for motif in motifs_type_demande:
        match = re.search(motif, question)
        if match:
            return {
                "type": match.group(0).strip(),
                "statut": "inconnu"}
        
    # Si aucun type précis
    return {"type": None, "statut": "aucun"}


# Recherche des évènements pertinents en fonction de l'actualité de l'évènement au moment de l'envoi du prompt
def recherche_event_pertinent(query, k = 300, max_results = 20):
    index, metadata = load_faiss_index()

    print("FAISS :", index.ntotal)
    print("Metadata :", len(metadata))

    q_emb = embed_query(query) # transforme le prompt en numérique
    distances, indices = index.search(np.array(q_emb, dtype="float32").reshape(1, -1), k) # transforme la liste de floats en tableau numpy avec vecteurs et dimension

    intervalle = extraire_intervalle_temporel(query)

    ville = extraire_ville(query)
    print("VILLE EXTRAITE :", ville)
    print("INTERVALLE :", intervalle)

    # Vérifier si la ville extraite existe dans les données
    if ville is not None:
        villes_disponibles = {normalize_city(event.get("city"))
            for event in metadata
            if normalize_city(event.get("city")) is not None}

        if ville not in villes_disponibles:
            print(
                f"VILLE '{ville}' NON PRÉSENTE DANS LA BASE "
                "-> aucun événement")
            return []

    # extraction âge
    m = re.search(r"(\d+)\s*ans", query.lower())
    age_demande = int(m.group(1)) if m else None

    # Type évènement
    type_info = extraire_type_evenement(query)

    type_evenement = type_info["type"]
    type_statut = type_info["statut"]

    print("TYPE ÉVÈNEMENT :", type_evenement)
    print("STATUT TYPE :", type_statut)

    if type_statut == "inconnu":
        print(
            f"Type/domaine demandé non pris en charge : "
            f"{type_evenement}"
        )
        return []

    results = []
    uid_vu = set()

    nb_actif = 0
    nb_ville = 0
    nb_date = 0
    nb_age = 0
    nb_event_type = 0

    print(indices.shape)
    print(indices[0][:20])

    for idx in indices[0]:
        print(idx)
        event = metadata[idx]

        print("\n---------------------")
        print("\nEvent:", event["title"])

        # éviter doublons
        if event["uid"] in uid_vu:
            print("REJET uid")
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
                pass
            # si seul l'âge minimum est renseigné
            elif age_min_event is not None and age_demande < age_min_event:
                continue

            # si seul l'âge maximum est renseigné
            elif age_max_event is not None and age_demande > age_max_event:
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

        # filtre type d'évènement
        if type_evenement is not None:
            print("\n========== DEBUG TYPE ==========")
            print("TYPE DEMANDÉ :", repr(type_evenement))

            type_event = determiner_type_evenement(event)

            print("UID :", event.get("uid"))
            print("TITLE :", repr(event.get("title")))
            print("TITLE_FR :", repr(event.get("title_fr")))
            print("DESCRIPTION :", repr(event.get("description")))
            print("DESCRIPTION_FR :", repr(event.get("description_fr")))
            print("LONGDESCRIPTION :", repr(event.get("longdescription")))
            print("LONGDESCRIPTION_FR :", repr(event.get("longdescription_fr")))
            print("KEYWORDS_FR :", repr(event.get("keywords_fr")))
            print("TYPE DÉTECTÉ :", repr(type_event))
            print("================================")

            print("Type demandé :", type_evenement)
            print("Type détecté événement :", type_event)

            if type_event is None or type_evenement not in type_event:
                print("-> rejet type")
                continue

        print("-> type OK")
        nb_event_type += 1
           
        uid_vu.add(event["uid"])
        results.append(event)

        if len(results) >= max_results:
            break

    print("\n========== DEBUG ==========")
    print("Ville détectée :", ville)
    print("Intervalle :", intervalle)
    print("Après filtre actif :", nb_actif)
    print("Après filtre ville :", nb_ville)
    print("Après filtre âge :", nb_age)
    print("Après filtre temporel :", nb_date)
    print("Après filtre type event :", nb_event_type)
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
- Si le contexte est vide ou si la ville demandée n'est pas présente dans la région Grand Est: 
    -Tu dois répondre EXACTEMENT la phrase suivante, sans rien ajouter: "Je n'ai trouvé aucun événement correspondant à votre recherche dans la base de données."
    - Tu NE DOIS PAS proposer d’alternatives.
    - Tu NE DOIS PAS suggérer d’autres villes.
    - Tu NE DOIS PAS recommander d’autres événements.
    - Tu NE DOIS PAS ajouter d’explications.
    - Tu NE DOIS PAS reformuler la phrase.
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
# générer  une réponse à partir d'un contexte déjà construit
def generate_answer_from_context(question, context):
    if not context.strip():
        return "Je n'ai trouvé aucun événement correspondant à votre recherche."
    
    chain = prompt | chatbot_llm
    response = chain.invoke(
        {
            "context": context,
            "question": question
        })
    return response.content


def generate_answer(question):
    event = recherche_event_pertinent(question, max_results = 10)

    if len(event) == 0:
        return {"answer": "Je n'ai trouvé aucun événement correspondant à votre recherche.", "events": [], "context": ""}
    
    context = build_context(event)
    answer = generate_answer_from_context(question, context)

    return {"answer": answer, "events": event, "context": context}


# Classe pour API
class PulsEventRAG:
    def __init__(self):
        try:
        # On réutilise directement TON index FAISS et TES metadata
            self.index, self.metadata = self.load_index()
        except Exception as e:
            print("Erreur dans __init__ :", e)
            self.index, self.metadata = None, None

    # 1. Extraction
    def _extract(self):
        print("Début extraction", flush=True)
        URL = ("https://public.opendatasoft.com/api/explore/v2.1/"
            "catalog/datasets/evenements-publics-openagenda/records")

        PERIODES = [
            ("2025-01-01T00:00:00+00:00", "2025-04-01T00:00:00+00:00"),
            ("2025-04-01T00:00:00+00:00", "2025-07-01T00:00:00+00:00"),
            ("2025-07-01T00:00:00+00:00", "2025-10-01T00:00:00+00:00"),
            ("2025-10-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
            ("2026-01-01T00:00:00+00:00", "2026-04-01T00:00:00+00:00"),
            ("2026-04-01T00:00:00+00:00", "2026-07-01T00:00:00+00:00"),
            ("2026-07-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00"),
            ("2026-10-01T00:00:00+00:00", "2027-01-01T00:00:00+00:00"),]

        LIMIT = 100
        all_events = []

        for debut, fin in PERIODES:
            print(f"Période : {debut} -> {fin}", flush=True)
            params = {
                "refine": ["location_region:Grand Est"],
                "where": (
                    f'lastdate_begin >= "{debut}" '
                    f'AND firstdate_begin < "{fin}"'
                ),
                "limit": LIMIT,
                "offset": 0,}

            response = requests.get(URL, params=params)
            response.raise_for_status()
            data = response.json()
            total_count = data["total_count"]
            print("Total :", total_count, flush=True)

            periode_events = []
            for offset in range(0, total_count, LIMIT):
                params["offset"] = offset
                response = requests.get(URL, params=params)
                response.raise_for_status()
                page = response.json()["results"]
                periode_events.extend(page)

            all_events.extend(periode_events)

        # Suppression des doublons
        unique = {e["uid"]: e for e in all_events}
        return list(unique.values())   

    # 2. Preprocessing 
    def _preprocess(self, events):
        # Evènements à exclure
        agenda_a_exclure = [
            "Mes événements France Travail",
            "France-Belgique - Calendrier des évènements économiques et sectoriels",
            "Ensemble, dialoguons - Édition 2026 | Banque de France",
            "Ambassadeurs IA",
            "Chambre d'agriculture de la Moselle",
            "Chambre d'agriculture des Vosges",
            "Chambre d'agriculture de la Meurthe-et-Moselle",
            "Chambre d'agriculture Grand-Est",
            "Chambre d'agriculture de la Meuse"]

        # Filtres des évènements à exclure
        filtered = []
        for e in events:
            origin = e.get("originagenda_title") or ""
            if "Archive" in origin:
                continue
            if origin in agenda_a_exclure:
                continue
            filtered.append(e)

        # Champs à conserver
        champs_a_garder = ["uid", "canonicalurl", "title_fr", "description_fr", "longdescription_fr", "keywords_fr", "conditions_fr", "timings",
                            "daterange_fr", "firstdate_begin", "lastdate_end", "location_name", "location_address", 
                            "location_postalcode", "location_city", "location_department", "location_region", "age_min", "age_max", "registration"]
        today = datetime.now(timezone.utc)
        final_events = []

        for e in filtered:
            new_e = {c: e.get(c) for c in champs_a_garder}
            new_e["type_evenement"] = determiner_type_evenement(new_e)

            # Statut actif
            last_date = e.get("lastdate_end")
            if last_date:
                try:
                    date_fin = datetime.fromisoformat(last_date)
                    new_e["event_actif"] = date_fin >= today
                except Exception:
                    new_e["event_actif"] = False
            else:
                new_e["event_actif"] = False

            # Construction du texte RAG
            parts = [
                    f"Titre: {new_e['title_fr']}",
                    f"Description: {new_e['description_fr']}",
                    f"Longue description: {new_e['longdescription_fr']}",
                    f"Conditions: {new_e['conditions_fr']}",
                    f"Lieu: {new_e['location_name']}",
                    f"Age minimum: {new_e['age_min']}",
                    f"Age maximum: {new_e['age_max']}",
                    f"Ville: {new_e['location_city']}",
                    f"Adresse: {new_e['location_address']}",
                    f"Code postal: {new_e['location_postalcode']}",
                    f"Département: {new_e['location_department']}",
                    f"Région: {new_e['location_region']}",
                    f"Periodes: {new_e['timings']}",
                    f"Date: {new_e['daterange_fr']}",
                    f"Firstdate_debut: {new_e['firstdate_begin']}",
                    f"Lastdate_fin: {new_e['lastdate_end']}",
                    f"Lien: {new_e['canonicalurl']}",
                    f"Type_evenement: {new_e['type_evenement']}"]
            texte_rag = "\n".join([p for p in parts if p])
            new_e["texte_rag"] = texte_rag

            final_events.append(new_e)
        return final_events

    # 3. Chunking
    def _chunking(self, events):
        # Initialisation du splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""])
    
        chunks = []
        metadata = []

        for e in events:
            texte = e.get("texte_rag", "")
            if not texte:
                continue

            # Extraire timing_begin / timing_end
            raw_timings = e.get("timings")
            timing_begin, timing_end = None, None

            if raw_timings:
                try:
                    timings = json.loads(raw_timings) if isinstance(raw_timings, str) else raw_timings
                    if timings and isinstance(timings, list):
                        t0 = timings[0]
                        timing_begin = t0.get("begin")
                        timing_end = t0.get("end")
                except Exception:
                    pass

            # Découpage du texte
            for ch in text_splitter.split_text(texte):
                chunks.append(ch)

                metadata.append({
                    "uid": e.get("uid"),
                    "title": e.get("title_fr"),
                    "keywords": e.get("keywords_fr"),
                    "description": e.get("description_fr"),
                    "longdescription": e.get("longdescription_fr"),
                    "city": e.get("location_city"),
                    "lieu": e.get("location_name"),
                    "date": e.get("daterange_fr"),
                    "timing_begin": timing_begin,
                    "timing_end": timing_end,
                    "firstdate_begin": e.get("firstdate_begin"),
                    "lastdate_end": e.get("lastdate_end"),
                    "conditions": e.get("conditions_fr"),
                    "age_minimum": e.get("age_min"),
                    "age_maximum": e.get("age_max"),
                    "canonicalurl": e.get("canonicalurl"),
                    "type_evenement": e.get("type_evenement"),
                    "chunk": ch
                })

        return chunks, metadata

    # 4. Embeding
    def _embeding(self, chunks):
        api_key = os.getenv("PULSEVENT_MISTRAL_KEY")
        if not api_key:
            raise ValueError("❌ Clé API Mistral manquante")

        client = Mistral(api_key=api_key)
        model = "mistral-embed"

        BATCH_SIZE = 18
        vectors = []

        # envoie de batch au modèle Mistral avec un temps de pause afin de gérer la limite Mistral
        def embed_batch(batch_texts):
            while True:
                try:
                    print(type(batch_texts))
                    print(len(batch_texts))
                    print(type(batch_texts[0]))
                    print(batch_texts[0][:200])
                    response = client.embeddings.create(
                        model=model,
                        inputs=batch_texts
                    )
                    return [item.embedding for item in response.data]

                except Exception as e:
                    print(repr(e))
                    if "429" in str(e):
                        print("Rate limit atteint <> pause 5 secondes…")
                        time.sleep(5)
                        continue
                    # else:
                    raise

        # Découpage en batch
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            print(f"Batch {i}/{len(chunks)}", flush=True)
            batch_texts = batch  # batch = liste de strings

            batch_vectors = embed_batch(batch_texts)
            vectors.extend(batch_vectors)

            # Pause légère pour éviter le 429
            time.sleep(3)

        # Conversion en tableau numpy pour FAISS
        return np.array(vectors, dtype="float32")    

    # 5. Indexation
    def _faiss_build(self, vectors):
        d = vectors.shape[1]  # dimension des embeddings
        index = faiss.IndexFlatL2(d)
        index.add(vectors)

        return index

    # Sauvegarde index FAISS & metadonnées
    def _faiss_saving(self, index, metadata):
        Path("faiss_index").mkdir(exist_ok=True)

        # Sauvegarde FAISS
        faiss.write_index(index, "./faiss_index/faiss.idx")

        # Sauvegarde metadata
        with open("./faiss_index/metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load_index(self):
        print("Chargement de l'index FAISS...")
        try:
            # return load_faiss_index()
            index = faiss.read_index("./faiss_index/faiss.idx")
            with open("./faiss_index/metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
            return index, metadata
        except Exception:
            return None, None


    def rebuild_index(self):
        print("===== DEBUT REBUILD =====", flush=True)

        print("Extraction...", flush=True)
        t0 = time.perf_counter()
        events = self._extract()
        print(f"{len(events)} événements", flush=True)
        print(f"Extraction: {time.perf_counter() - t0:.1f} s")

        print("Préprocessing...", flush=True)
        t1 = time.perf_counter()
        clean = self._preprocess(events)
        print(f"Taille data traité: {len(clean)}", flush=True)
        print(f"Préprocessing: {time.perf_counter() - t1:.1f} s")

        print("Chunking...", flush=True)
        t2 = time.perf_counter()
        chunks, metadata = self._chunking(clean)
        print(f"Taille chunks: {len(chunks)}", flush=True)
        print(f"Chunking : {time.perf_counter() - t2:.1f} s")

        print("Embedding...", flush=True)
        t3 = time.perf_counter()
        vectors = self._embeding(chunks)
        print(f"Taille embeddings: {len(vectors)}", flush=True)
        print(f"Embeddings : {time.perf_counter() - t3:.1f} s")

        print("FAISS...", flush=True)
        t4 = time.perf_counter()
        index = self._faiss_build(vectors)
        print(f"Taille index: {len(index)}", flush=True)
        print(f"Embeddings : {time.perf_counter() - t4:.1f} s")

        print("Saving...", flush=True)
        self._faiss_saving(index, metadata)

        self.index = index
        self.metadata = metadata

        print("===== FIN =====", flush=True)
        return "Index reconstruit avec succès."

    def ask(self, question: str) -> str:
        if self.index is None:
            return "⚠️ L’index n’est pas encore construit. Lancez /rebuild."
        return generate_answer(question)

# Tests
if __name__ == "__main__":
    print(generate_answer("Je cherche des ateliers à Charleville-Mézières."))
