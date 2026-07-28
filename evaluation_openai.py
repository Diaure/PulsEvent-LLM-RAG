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
    "Y a-t-il un concert à Strasbourg ?",
    "Je cherche un atelier pour un enfant de 8 ans à Reims.",
    "Quels événements sont prévus ce week-end à Metz ?",
    "Y a-t-il des événements gratuits à Nancy ?",
    "Que faire en famille à Mulhouse demain ?",
    "Je cherche des ateliers à Charleville-Mézières.",
    "Quels événements sont adaptés aux seniors à Colmar ?",
    "Je cherche une exposition le mois prochain à Strasbourg.",
    "Je cherche un événement culturel à Tokyo."
    "Je cherche un cours de plongée sous-marine à Reims."
]

ground_truths = [
    "",
    "Les ateliers adaptés à un enfant de 6 ans à Reims sont les ateliers culinaires pour enfants à La Belle Enchantée, l'atelier familial « La grande maquette » au FRAC Champagne-Ardenne et la visite « Les coulisses de la création » à La Comédie de Reims. La réponse doit préciser les dates, les conditions de participation, les tranches d'âge concernées et les liens des événements lorsque ces informations sont disponibles.",
    
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
