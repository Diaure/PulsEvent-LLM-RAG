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
    "Pour un enfant à Reims, j’ai repéré plusieurs activités adaptées: un atelier culinaire pour les 6–12 ans, un atelier “bacs de fouilles” accessible dès 8 ans, et une activité familiale autour de la grande maquette. Ce sont les options les plus adaptées aux enfants en ce moment.",
    "Ce week-end à Strasbourg, il y a un concert jazz au Palais Rohan, un atelier famille au Musée Alsacien, et une visite guidée de la Neustadt. Ce sont les activités les plus intéressantes pour cette période.",
    "Oui, à Metz il y a un concert gratuit de musique classique à l’Arsenal, ainsi qu’un concert en plein air au Parc de la Seille.",
    "À Nancy, plusieurs activités conviennent bien aux seniors: un atelier bien-être au Centre Jean Prouvé, une conférence sur le patrimoine au Musée Lorrain, et une balade culturelle dans la vieille ville. Ce sont des activités calmes et adaptées.",
    "Pour une sortie en famille à Mulhouse demain, vous pouvez faire un atelier créatif au Musée des Beaux-Arts, visiter le Parc Zoologique et Botanique, ou participer à une animation scientifique à la Cité du Train. Ce sont des activités adaptées à tous les âges.",
    "À Champagne et à Charleville‑Mézières, il y a plusieurs ateliers créatifs intéressants. On trouve notamment des ateliers d’aquarelle, des ateliers de dessin ou de BD, ainsi que des activités manuelles proposées dans les médiathèques ou les centres culturels. Ce sont les options les plus adaptées si vous cherchez une activité créative dans ces villes.",

]

# générer les réponses IA et les contextes
reponses = []
contextes = []

for q in questions:
    time.sleep(1.5) # delai entre les appels Mistral pour éviter d'atteindre la limte d'appel Mistral
    ia_answer = generate_answer(q)
    reponses.append(ia_answer)

    retrieved = recherche_event_pertinent(q)
    ctx = build_context(retrieved)
    contextes.append([ctx]) # pour chaque question, ragas attend une liste de string pour le contexte

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