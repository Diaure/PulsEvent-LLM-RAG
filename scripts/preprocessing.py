# Ce fichier est utilisé pour filtrer l'extraction uniquement sur les évènements souhaités
import json
from collections import Counter
import pandas as pd

# Lecture de l'extraction
with open("data/grand_est_events.json", "r", encoding="utf-8") as f:
    ge_events = json.load(f)
    # print(ge_events)

# liste_categ = []
# for c in ge_events:
#     categorie = c.get("category")
#     liste_categ.append(categorie)
# print(liste_categ) # None pour tous

# Compter les occurrences keywords
liste_kw = []
kws_compte = {}  

for e in ge_events:
    kws = e.get("keywords_fr") or []
    liste_kw.extend(kws)
kws_compte = Counter(liste_kw)  


# # # Compter les occurrences titre origine
liste_to = []
tos_compte = {}  

for e in ge_events:
    tos = e.get("originagenda_title") or []
    if tos:
        liste_to.append(tos)
tos_compte = Counter(liste_to)  

print("\n=== MOTS-CLÉS (keywords_fr) ===")
for kw, count in kws_compte.most_common():
    print(f"{kw}: {count}")

print("\n=== TITRES D'ORIGINE (originagenda_title) ===")
for to, count in tos_compte.most_common():
    print(f"{to}: {count}")

# Au final pour filtrer les évènements on passera par le titre original qui est plus information, comparé aux mots clés qui sont parfois nuls 

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

ge_events_data_filtre = []
for e in ge_events:
    origin = e.get("originagenda_title") or ""
    if "Archive" in origin: # ignorer les évènements "Archives"
        continue
    if origin in agenda_a_exclure: # ignorer les évènements non culturels
        continue
    if origin not in agenda_a_exclure:
        ge_events_data_filtre.append(e)
print(len(ge_events))
print(len(ge_events_data_filtre))

# Sauvegarde du fichier filtré
with open("data/ge_events_data_filtre.json", "w", encoding="utf-8") as f:
    json.dump(ge_events_data_filtre, f, ensure_ascii=False, indent=4)

# Les mots clés après filtre
liste_kws = []

for e in ge_events_data_filtre:
    kw = e.get("keywords_fr") or []
    liste_kws.extend(kw)

kws_compte = Counter(liste_kws)

# Compter les occurrences titre origine
liste_to = []
tos_compte = {}  

for e in ge_events_data_filtre:
    tos = e.get("originagenda_title") or []
    if tos:
        liste_to.append(tos)
tos_compte = Counter(liste_to)  

print("\n=== MOTS-CLÉS après filtre ===")
for cle, count in kws_compte.most_common():
    print(f"{cle}: {count}")

print("\n=== TITRES D'ORIGINE après filtre ===")
for to, count in tos_compte.most_common():
    print(f"{to}: {count}")


# Conserver uniquement les éléments nécessaires pour le chatbot(id event, titre, description, vile, lieu, dates debut et fin, lien de l'evenement pour plus d'information, 
# age min/max, date_rage-date complete et heure, )
champs_a_garder = [
    "uid", "canonicalurl",
    "title_fr", "description_fr", "longdescription_fr",
    "conditions_fr",
    "daterange_fr", "firstdate_begin", "lastdate_end",
    "location_name", "location_address", "location_postalcode", "location_city", "location_department", "location_region",
    "age_min", "age_max", "registration"]

ge_event_rag = []
nvel_element = {}
for e in ge_events_data_filtre:
    nvel_element = {c: e.get(c) for c in champs_a_garder} 
    ge_event_rag.append(nvel_element) 

print(len(ge_events_data_filtre))
print(len(ge_event_rag))

# Sauvegarde du fichier de base pour embeding
with open("data/ge_event_rag.json", "w", encoding="utf-8") as f:
    json.dump(ge_event_rag, f, ensure_ascii=False, indent=4)


# Dataframe + le champ texte pour le RAG
ge_events_df = pd.DataFrame(ge_event_rag)
def contruire_texte_rag(row):
    parts = [
        f"Titre: {row['title_fr']}",
        f"Description: {row['description_fr']}",
        f"Longue description: {row['longdescription_fr']}",
        f"Conditions: {row['conditions_fr']}",
        f"Lieu: {row['location_name']}",
        f"Age minimum: {row['age_min']}",
        f"Age maximum: {row['age_max']}",
        f"Ville: {row['location_city']}",
        f"Adresse: {row['location_address']}",
        f"Code postal: {row['location_postalcode']}",
        f"Département: {row['location_department']}",
        f"Région: {row['location_region']}",
        f"Date: {row['daterange_fr']}",
        f"Début: {row['firstdate_begin']}",
        f"Fin: {row['lastdate_end']}",
        f"Lien: {row['canonicalurl']}"
    ]
    return "\n".join([p for p in parts if p])
ge_events_df["texte_rag"] = ge_events_df.apply(contruire_texte_rag, axis=1)
print(len(ge_events_df))
print(ge_events_df.columns)
print(ge_events_df["texte_rag"].iloc[0])
print(ge_events_df["registration"].iloc[6])
print(ge_events_df.head(15))

ge_events_df.to_csv("data/ge_events_df.csv")

