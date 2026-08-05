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
from chatbot_rag import generate_answer_from_context, recherche_event_pertinent, build_context

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
    "Que faire en famille à Mulhouse ce week-end ?",
    "Je cherche des ateliers à Charleville-Mézières.",
    "Quels événements sont adaptés aux seniors à Colmar ?",
    "Je cherche une exposition le mois prochain à Strasbourg.",
    "Je cherche un événement culturel à Tokyo.",
    "Je cherche un cours de plongée sous-marine à Reims."
]

ground_truths = [
    "Plusieurs concerts sont disponibles à Strasbourg. Parmi les plus pertinents figurent: - Concert de musique classique à Strasbourg – Ensemble Musicâme France - Vêpres italiennes des XVIème et XVIIème siècles par l'Ensemble Triosa. Pour chacun des évènements, les dates, les conditions de participation et les informations pratiques sont indiquées lorsqu'elles sont disponibles",
    "Voici des ateliers adaptés à un enfant de 8 ans à Reims: - Atelier 'bacs de fouilles' pour les enfants - Ateliers culinaires pour enfants à la Belle Enchantée - Les coulisses de la création. Pour chacun, les dates, les conditions de participation, les tranches d'âge concernées, et les liens des événements lorsque disponibles sont clairement mentionnés.",
    "Aucun événement correspondant à cette recherche n'est présent dans la base de données. La réponse n'invente pas d'événement, ni ne propose des activités non pertinentes.",
    "Plusieurs évènements gratuits à Nancy sont présent dans la base de données parmis lesquels: - Exposition Présents - Visite de Nancy Thermal : un projet né il y a 117 ans - Escape Game 'Panique en cabine'. Les dates, les conditions d'accès, la gratuité, ainsi que les informations complémentaires utiles pour chaque évènement sont clairement indiqués si disponibles.",
    "Aucun évènement correspondant à cette recherche n'est disponible dans la base. Si aucun évènement, la réponse l'indique clairement sans proposer d'évènements hors du contexte demandé.",
    "Les évènements de type 'atelier'' à Charleville-Mézières présents dans la base sont - 'Nuée' (spectacle et atelier autour de la marionnette) - Les Quiquoi et le chien moche dont personne ne veut. Pour chaque atelier, la réponse précise les dates, les conditions ainsi que toutes informations pratiques lorsque disponibles.",
    "Les événements adaptés aux seniors à Colmar retrouvés dans la base de données - '4026 Des jouets éternels ?' - Visite guidée de la Comédie de Colmar - Visite du bâtiment administratif du XIXe siècle. La réponse indique pour chacun des évènements les dates, les conditions ainsi que toutes informations pratiques lorsque disponibles.",
    "Plusieurs expositions disponibles à Strasbourg le mois prochain dan la base de données, dont: - Visite libre de l’exposition « Un Voyage à Strasbourg » - Archifoto 2026 - Visite guidée de l'exposition 'Un voyage à Strasbourg'. Pour chacune des expositions, les dates, les liens, les conditions sont précisées",
    "Aucun événement correspondant n'est disponible dans la base. La réponse l'indiquer clairement sans proposer d'événements situés dans une autre ville.",
    "Aucun cours de plongée sous-marine n'a été trouvé à Reims dans la base de données. La réponse n'invente aucun événement, ni ne propose d'autres activités."
]


# Génération des réponses IA + contextes
answers = []
contexts = []

for question in questions_test:
    # récupérer l'évènement
    retrieved = recherche_event_pertinent(question, max_results=6)

    # construire le contexte tel que dans les données
    context = build_context(retrieved)

    # réponse générée
    reponse_ia = generate_answer_from_context(question, context)
    answers.append(reponse_ia)

    # contexte évalué par RAGAS
    contexts.append([context])
    # current_context = []
    # for event in retrieved:
    #     event_text = f"""
    #     Titre : {event['title']}
    #     Ville : {event['city']}
    #     Date : {event['date']}
    #     Début : {event['timing_begin']}
    #     Fin : {event['timing_end']}
    #     Conditions : {event['conditions']}
    #     Age minimum : {event['age_minimum']}
    #     Age maximum : {event['age_maximum']}
    #     Description : {event['chunk']}
    #     """
    #     current_context.append(event_text)
    

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
results_df.to_csv("data/ragas_results.csv", index=False)

print(results_df)
print(results_df.mean(numeric_only=True))
print(results_df["answer_relevancy"])
print(results_df.columns)
print(results_df[["user_input","response","answer_relevancy"]])

for i, a in enumerate(answers):
    print(i, type(a), repr(a[:100]))