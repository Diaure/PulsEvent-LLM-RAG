import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from chatbot_rag import generate_answer, recherche_event_pertinent, build_context

# Configuration
load_dotenv()
api_key = os.getenv("PULSEVENT_MISTRAL_KEY")
ragas_llm = ChatMistralAI(model="mistral-small-latest", api_key=api_key)
ragas_embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)


# Questions tests & réponses attendues
questions_test = [
    "Y a-t-il un concert à Strasbourg ?",
    "Je cherche un atelier pour un enfant de 8 ans à Reims.",
    "Quels événements sont prévus ce week-end à Metz ?",
    "Y a-t-il des événements gratuits à Nancy ?",
    "Que faire en famille à Mulhouse demain ?",
    "Je cherche des ateliers à Charleville-Mézières.",
    "Quels événements sont adaptés aux seniors à Colmar ?",
    "Je cherche une exposition le mois prochain à Strasbourg.",
    "Je cherche un événement culturel à Tokyo.",
    "Je cherche un cours de plongée sous-marine à Reims."
]

ground_truths = [
    "Oui. Les concerts de musique prévus à Strasbourg très prochainement sont le «Concert de musique classique à Strasbourg – Ensemble Musicâme France » et « Vêpres italiennes des XVIᵉ et XVIIᵉ siècles par l'Ensemble Triosa ». La réponse doit préciser les dates, les conditions de participation, les tranches d'âge concernées et les liens des événements lorsque ces informations sont disponibles.",
    "Les ateliers adaptés à un enfant de 6 ans à Reims sont les ateliers culinaires pour enfants à La Belle Enchantée, l'atelier familial « La grande maquette » au FRAC Champagne-Ardenne et la visite « Les coulisses de la création » à La Comédie de Reims. La réponse doit préciser les dates, les conditions de participation, les tranches d'âge concernées et les liens des événements lorsque ces informations sont disponibles.",
    "Aucun événement correspondant à la recherche d'événements prévus ce week-end à Metz n'a été trouvé dans la base de données.",
    "Oui, il y a des évènements gratuits à Nancy. Les événements gratuits proposés à Nancy sont l'exposition « Présents » à la Galerie Poirel et « Le campus Carnot se dévoile ». La réponse doit mentionner les dates, les conditions d'accès, la gratuité des événements ainsi que les informations complémentaires utiles.",
    "Malheureusement, aucune activité familiale correspond à la recherche à Mulhouse pour la période souhaitée n'a été trouvée dans la base de données.",
    "Les événements les plus pertinents trouvés à Charleville-Mézières comprennent des visites guidées, un marché artisanal et des découvertes patrimoniales. La réponse doit préciser qu'aucun atelier créatif explicite n'est présent dans la base de données et proposer uniquement les événements les plus proches de la requête.",
    "Les événements adaptés à un public senior à Colmar sont l'exposition « 4026 Des jouets éternels ? », l'ouverture exceptionnelle de la Cour d'appel de Colmar, la visite guidée de la Comédie de Colmar et la visite du bâtiment administratif du XIXe siècle. La réponse doit expliquer en quoi ces activités sont adaptées aux seniors et fournir leurs principales caractéristiques.",
    "Aucune exposition correspondant aux critères temporels et géographiques de la recherche à Strasbourg n'a été trouvée dans la base de données.",
    "Aucun événement culturel n'est disponible à Tokyo dans la base de données. La réponse doit uniquement indiquer l'absence de résultat et ne doit proposer aucun événement situé dans une autre ville ou région.",
    "Aucun cours de plongée sous-marine n'a été trouvé à Reims dans la base de données. La réponse doit indiquer clairement qu'aucun événement correspondant n'est disponible."
]

# Génération des réponses IA + contextes

answers = []
contexts = []

for question in questions_test:
    # récupérer l'évènement
    retrieved = recherche_event_pertinent(question, max_results=6)

    # réponse générée
    reponse_ia = generate_answer(question)
    answers.append(reponse_ia)

    # contexte au format RAGAS
    current_context = []
    for event in retrieved:
        event_text = f"""
        Titre : {event['title']}
        Ville : {event['city']}
        Date : {event['date']}
        Début : {event['timing_begin']}
        Fin : {event['timing_end']}
        Conditions : {event['conditions']}
        Age minimum : {event['age_minimum']}
        Age maximum : {event['age_maximum']}
        Description : {event['chunk']}
        """
        current_context.append(event_text)
    contexts.append(current_context)

for i, answer in enumerate(answers):
    print(f"Question {i+1}")
    print(answer)
    print("------------------")

# Dataset RAGAS
evaluation_dataset = Dataset.from_dict({
    "question": questions_test,
    "answer": answers,
    "contexts": contexts,
    "reference": ground_truths
})

# Évaluation RAGAS
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
print(results_df["answer_relevancy"])
print(results_df.columns())