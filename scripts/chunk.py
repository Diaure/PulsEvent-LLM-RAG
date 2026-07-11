import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle

# Chargment des données
ge_events_df = pd.read_csv(".\data\ge_events_df.csv")
print("Nombre d'évènements:", ge_events_df.shape[0])

print(ge_events_df["texte_rag"].str.len().describe())
print((ge_events_df["texte_rag"].str.len() > 700).sum())


# Code cours
# Découpage récursif avec chevauchement avec Langchain
# Initialisation du splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, # Taille de chaque segment en caractères <> Plus c’est grand, plus le chunk contient de contexte, mais les embeddings peuvent être moins précis.
    chunk_overlap=200, # Chevauchement entre les segments
    separators=["\n\n", "\n", ".", " ", ""]) # séparation pas faite au hasard des lignes(\n), espaces(" "), milieu des mots(""), paragraphes(\n\n)
    
# Découpage du texte
chunks = []

for idx, row in ge_events_df.iterrows():
    for chunk in text_splitter.split_text(row["texte_rag"]):
        chunks.append({
            "uid": row["uid"],
            "title": row["title_fr"],
            "city": row["location_city"],
            "date": row["daterange_fr"],
            "canonicalurl": row["canonicalurl"],
            "chunk": chunk})

print("Nombre total de chun:", len(chunk))

# Sauvegarde des chunks : les chunks sont une liste de dictionnaires python lourd, dump est le plus pratique pour conserver un fichier aussi lourd
with open("./data/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

