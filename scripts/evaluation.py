import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,)

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
import time

# Import de mon chatbot
from chatbot_rag import generate_answer, recherche_event_pertinent, build_context

load_dotenv()
api_key = os.getenv("PULSEVENT_MISTRAL_KEY")

# Liste de questions
questions = [
    "Je cherche un atelier pour un enfant à Reims.",
    "Quels événements sont prévus ce week-end à Strasbourg?",
    "Y a-t-il des concerts gratuits à Metz?",
    "Quels événements sont adaptés aux seniors à Nancy?",
    "Que faire en famille à Mulhouse demain?",
    "Je cherche des ateliers créatifs à Champagne et Charleville-Mézières."]

# Liste des réponses idéales
ground_truths = [
    "À Reims, trois ateliers sont adaptés aux enfants cette semaine : un atelier culinaire pour les 6–12 ans, un atelier “bacs de fouilles” accessible dès 8 ans, et un atelier créatif “La grande maquette” au FRAC. Ce sont les activités spécifiquement conçues pour les enfants.",
    "Ce week-end à Strasbourg, les activités principales sont un concert jazz au Palais Rohan, un atelier famille au Musée Alsacien et une visite guidée de la Neustadt. Ce sont les événements culturels programmés pour cette période.",
    "À Metz, il y a deux concerts gratuits : un concert de musique classique à l’Arsenal et un concert en plein air au Parc de la Seille. Ce sont les événements gratuits disponibles.",
    "À Nancy, trois activités conviennent aux seniors : une visite patrimoniale universitaire, une visite Art Nouveau au Parc de Saurupt et une visite des salles patrimoniales de la Chambre de Commerce. Ce sont des activités adaptées à un public senior.",
    "À Mulhouse, trois activités familiales sont proposées demain : un atelier créatif au Musée des Beaux-Arts, une visite du Parc Zoologique et Botanique et une animation scientifique à la Cité du Train. Ce sont les activités adaptées à tous les âges.",
    "À Charleville-Mézières et dans la région de Champagne, plusieurs ateliers créatifs sont proposés : aquarelle, dessin, BD et activités manuelles dans les médiathèques. Ce sont les ateliers créatifs disponibles.",
]

# générer les réponses IA et les contextes
reponses = []
contextes = []

# def split_text(text, chunk_size=300):
#     words = text.split()
#     chunks = []
#     current = []

#     for w in words:
#         current.append(w)
#         if len(current) >= chunk_size:
#             chunks.append(" ".join(current))
#             current = []

#     if current:
#         chunks.append(" ".join(current))

#     return chunks


for q in questions:
    time.sleep(1.5) # delai entre les appels Mistral pour éviter d'atteindre la limte d'appel Mistral
    ia_answer = generate_answer(q)
    reponses.append(ia_answer)

    retrieved = recherche_event_pertinent(q)
    chunks = [r["chunk"] for r in retrieved]
    contextes.append(chunks)

    # ctx = build_context(retrieved)
    # chunks = split_text(ctx, chunk_size=300)
    # contextes.append(chunks) # pour chaque question, ragas attend une liste de string pour le contexte
    print(contextes[0])

# construire le dataset complet pour RAGAS
evaluation_data = {
    "question": questions,
    "answer": reponses,
    "contexts": contextes,
    "ground_truth": ground_truths}

evaluation_df = Dataset.from_dict(evaluation_data)
print("Dataset d'évaluation prêt.")

# Configuration LLM & Embeddings
mistral_llm = ChatMistralAI(mistral_api_key = api_key, model = "mistral-large-latest", temperature=0.1)
mistral_embeddings = MistralAIEmbeddings(mistral_api_key= api_key, model = "mistral-embed")

# Définition des métriques à calculer
metrics_to_evaluate = [
        faithfulness,       # Génération: fidèle au contexte ?
        answer_relevancy,   # Génération: réponse pertinente à la question ?
        context_precision,  # Récupération: contexte précis (peu de bruit) ?
        context_recall,]     # Récupération: infos clés récupérées (nécessite ground_truth) ?
print(f"Métriques sélectionnées: {[m.name for m in metrics_to_evaluate]}")

# Lancement de l'évaluation Ragas
print("\nLancement de l'évaluation Ragas (peut prendre du temps)...")
time.sleep(2)
results = evaluate(
        dataset = evaluation_df,
        metrics = metrics_to_evaluate,
        llm = mistral_llm,                # LLM pour juger certaines métriques
        embeddings = mistral_embeddings)   # Embeddings pour juger d'autres métriques
print("\n--- Évaluation Ragas terminée ---")

# Affichage des résultats
results_df = results.to_pandas()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n--- Résultats RAGAS ---")
print(results_df)

print("\n--- Scores moyens ---")
print(results_df.mean(numeric_only=True))

# Export CSV
output_path = "data/ragas_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\nFichier CSV exporté : {output_path}")