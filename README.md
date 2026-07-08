# Puls Event Intelligent Chatbot

## Extraction des données OpenAgenda

### Source des données

Les données proviennent du jeu de données OpenAgenda disponible sur OpenDataSoft :

`https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda`

API utilisée :

`https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records`

### Objectif

Afin d'alimenter le système RAG, les événements ont été extraits depuis le jeu de données **Événements publics OpenAgenda** disponible sur l'API ***OpenDataSoft***.

L'extraction a été limitée:
- à la région **Grand Est**
- aux évènements compris entre le 1er janvier 2025 et le 31 décembre 2026

### Principes
L'API OpenDataSoft impose une limite: `offset + limit <= 10000`.

Dans ce contexte, il est impossible de récupérer l'ensemble des évènements sur une seule période de deux ans avec une seule requête. Pour contourner cette limitation, nous avons découper la période d'extraction en plusieurs intervalles chronologiques:

`
01/01/2025 - 01/04/2025,
01/04/2025 - 01/07/2025,
01/07/2025 - 01/10/2025,
01/10/2025 - 01/01/2026,
01/01/2026 - 01/04/2026,
01/04/2026 - 01/07/2026,
01/07/2026 - 01/10/2026,
01/10/2026 - 01/01/2027` 

Chaque période est alors paginée avec:
- `limit = 100`
- incrémentation de `offset` jusqu'au total d'évènements récupérés.

Les résultats de chaque période sont ensuite fusionnés dans une seule liste.

### Suppression des doublons

Une fois toutes les périodes récupérées, les événements sont dédupliquées grâce à leur identifiant unique (*uid*).

Cette étape garantit qu'un même événement présent sur plusieurs périodes n'apparaît qu'une seule fois dans le fichier final.

### Lancement du script

Depuis la racine du projet: `python scripts/extract.py`

Le fichier généré est enregistré dans: `data/grand_est_events.json`.














* il conserve la structure complète de chaque événement
* chaque événement peut être traité comme un document indépendant
* les métadonnées (titre, description, dates, lieu, mots-clés, lien, etc.) sont préservées et pourront être utilisées lors de l'indexation et de la recherche documentaire
* il est directement exploitable avec les bibliothèques Python utilisées pour le prétraitement des données et la création de la base vectorielle.

### Prétraitement

Après l'export, un prétraitement sera réalisé afin de nettoyer les données et de sélectionner les événements les plus pertinents pour le chatbot. Cette étape pourra notamment inclure:

* la gestion des valeurs manquantes
* le filtrage des événements hors du périmètre fonctionnel si nécessaire
* la préparation des documents avant leur vectorisation.

Ce corpus constituera ensuite la base documentaire interrogée par le système RAG lors des recherches effectuées par le chatbot.
