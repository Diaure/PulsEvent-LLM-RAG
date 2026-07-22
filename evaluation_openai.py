import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

from scripts.chatbot_rag import generate_answer, recherche_event_pertinent, build_context

load_dotenv()

# ============================
# 1. Modèle HuggingFace causal LM
# ============================

model_name = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=False
)

ragas_llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Embeddings HF compatibles RAGAS
ragas_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================
# 2. Jeu de tests
# ============================

questions_test = [
    "Je cherche un atelier pour un enfant à Reims.",
    "Quels événements sont prévus ce week-end à Strasbourg ?",
    "Y a-t-il des concerts gratuits à Metz ?",
    "Quels événements sont adaptés aux seniors à Nancy ?",
    "Que faire en famille à Mulhouse demain ?",
    "Je cherche des ateliers créatifs à Champagne et Charleville-Mézières."
]

ground_truths = [
    "Pour un enfant à Reims, j’ai repéré plusieurs activités adaptées : un atelier culinaire pour les 6–12 ans, un atelier “bacs de fouilles” accessible dès 8 ans, et une activité familiale autour de la grande maquette.",
    "Ce week-end à Strasbourg, il y a un concert jazz au Palais Rohan, un atelier famille au Musée Alsacien, et une visite guidée de la Neustadt.",
    "Oui, à Metz il y a un concert gratuit de musique classique à l’Arsenal, ainsi qu’un concert en plein air au Parc de la Seille.",
    "À Nancy, plusieurs activités conviennent bien aux seniors : un atelier bien-être au Centre Jean Prouvé, une conférence sur le patrimoine au Musée Lorrain, et une balade culturelle dans la vieille ville.",
    "Pour une sortie en famille à Mulhouse demain, vous pouvez faire un atelier créatif au Musée des Beaux-Arts, visiter le Parc Zoologique et Botanique, ou participer à une animation scientifique à la Cité du Train.",
    "À Champagne et à Charleville-Mézières, il y a plusieurs ateliers créatifs intéressants : aquarelle, dessin, BD, et des activités manuelles dans les médiathèques."
]

# ============================
# 3. Génération des réponses IA + contextes
# ============================

answers = []
contexts = []

for q in questions_test:
    ia_answer = generate_answer(q)
    answers.append(ia_answer)

    retrieved = recherche_event_pertinent(q)
    ctx_text = build_context(retrieved)
    contexts.append([ctx_text])

# ============================
# 4. Dataset RAGAS
# ============================

evaluation_data = {
    "question": questions_test,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

evaluation_dataset = Dataset.from_dict(evaluation_data)

# ============================
# 5. Évaluation RAGAS
# ============================

metrics_to_evaluate = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

results = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics_to_evaluate,
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

results_df = results.to_pandas()
results_df.to_csv("ragas_results.csv", index=False)

print(results_df)
print(results_df.mean(numeric_only=True))
