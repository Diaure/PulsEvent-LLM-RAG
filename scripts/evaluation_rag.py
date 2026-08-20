import os
import pandas as pd
# from datasets import Dataset
from dotenv import load_dotenv
import asyncio

# from ragas import evaluate
# from ragas.metrics.collections import (
#     faithfulness,
#     answer_relevancy,
#     context_precision,
#     context_recall,
# )

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# from langchain_mistralai import MistralAIEmbeddings
# from mistralai.client import Mistral
from openai import AsyncOpenAI
from ragas.llms import llm_factory
# from ragas.embeddings.base import embedding_factory
from ragas.embeddings import HuggingFaceEmbeddings
# from ragas.llms import LangchainLLMWrapper
from chatbot_rag import generate_answer_from_context, recherche_event_pertinent, build_context

# Configuration
load_dotenv()
api_key = os.getenv("PULSEVENT_MISTRAL_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("PULSEVENT_MISTRAL_KEY n'est pas définie.")

# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY n'est pas définie dans le .env")

# RAGAS LLM
# mistral_client = Mistral(api_key=api_key)
mistral_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1",
)


ragas_llm = llm_factory(
    "mistral-small-latest",
    client=mistral_client,)

# ragas_llm = LangchainLLMWrapper(
#     ChatMistralAI(
#         model="mistral-small-latest",
#         api_key=api_key,
#     )
# )


# RAGAS em embeddings
# openai_client = AsyncOpenAI(api_key=openai_api_key,)

# ragas_embeddings = embedding_factory(
#     "openai",
#     model="text-embedding-3-small", 
#     client=openai_client)

ragas_embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    normalize_embeddings=True,)

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

# for i, answer in enumerate(answers):
#     print(f"Question {i+1}")
#     print(answer)
#     print("------------------")

# # Dataset RAGAS
# evaluation_dataset = Dataset.from_dict({
#     "question": questions_test,
#     "answer": answers,
#     "contexts": contexts,
#     "ground_truth": ground_truths
# })

# Évaluation RAGAS
# metrics_to_evaluate = [
#     Faithfulness,
#     AnswerRelevancy,
#     ContextPrecision,
#     ContextRecall,
# ]

# metrics_to_evaluate = [
#     Faithfulness(llm=ragas_llm),
#     AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
#     ContextPrecision(llm=ragas_llm),
#     ContextRecall(llm=ragas_llm),
# ]


# Métriques
metrics_to_evaluate = {
    "faithfulness": Faithfulness(
        llm=ragas_llm
    ),
    "answer_relevancy": AnswerRelevancy(
        llm=ragas_llm,
        embeddings=ragas_embeddings
    ),
    "context_precision": ContextPrecision(
        llm=ragas_llm
    ),
    "context_recall": ContextRecall(
        llm=ragas_llm
    ),
}

# Debug
print("LLM:", ragas_llm)
print("EMBEDDINGS:", ragas_embeddings)
print("QUESTIONS:", len(questions_test))
print("ANSWERS:", len(answers))
print("CONTEXTS:", len(contexts))


async def run_evaluation():

    results = []

    for i, question in enumerate(questions_test):

        print(f"\n===== QUESTION {i + 1}/{len(questions_test)} =====")

        answer = answers[i]
        retrieved_contexts = contexts[i]
        reference = ground_truths[i]

        row = {
            "question": question,
            "answer": answer,
            "reference": reference,
        }

        result = await metrics_to_evaluate["faithfulness"].ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
        )
        row["faithfulness"] = result.value

        result = await metrics_to_evaluate["answer_relevancy"].ascore(
            user_input=question,
            response=answer,
        )
        row["answer_relevancy"] = result.value

        result = await metrics_to_evaluate["context_precision"].ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        )
        row["context_precision"] = result.value

        result = await metrics_to_evaluate["context_recall"].ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        )
        row["context_recall"] = result.value

        results.append(row)

        print(
            f"Faithfulness      : {row['faithfulness']}"
        )
        print(
            f"Answer relevancy  : {row['answer_relevancy']}"
        )
        print(
            f"Context precision : {row['context_precision']}"
        )
        print(
            f"Context recall    : {row['context_recall']}"
        )

    return results


results = asyncio.run(run_evaluation())

# results = evaluate(
#     dataset=evaluation_dataset,
#     metrics=metrics_to_evaluate,
# )

results_df = pd.DataFrame(results)

print("\n==============================")
print("RÉSULTATS")
print("==============================")
print(results_df)

print("\n==============================")
print("MOYENNES")
print("==============================")
metric_columns = ["faithfulness", "answer_relevancy", "context_precision", "context_recall",]
print(results_df[metric_columns].mean())

results_df.to_csv("data/ragas_results.csv", index=False)

print(results_df.mean(numeric_only=True))
if "answer_relevancy" in results_df.columns:
    print(results_df["answer_relevancy"])
print(results_df.columns)
print(results_df[["question","answer","answer_relevancy"]])

for i, a in enumerate(answers):
    print(i, type(a), repr(a[:100]))