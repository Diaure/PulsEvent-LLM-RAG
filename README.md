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

## Indexation - FAISS (Preprocessing > Chunking > Embedding)
### Preprocessing - Nettoyage et construction du texte RAG

Le script `preprocessing.py`:
- Charge les évènements bruts: `data/grand_est_events.json` 
- Analyse les mots-clés et titres pour identifier les sources non pertinentes
- Filtre les évènements (archives, évènements institutionnels, etc.)
- Conserve uniquement les champs utiles au chatbot (titre, description, lieu, dates, âge, lien…)
- Ajoute un champ event_actif basé sur la date de fin
- Construit un champ textuel complet texte_rag utilisé pour le RAG
- Exporte les données nettoyées au format JSON + CSV: 

`data/ge_event_rag.json, data/ge_events_df.csv`.

### Chunking - Découpage des textes
Le script `chunk.py` quant à lui:

- Récupère le dataframe des évènements filtrés précédemment `data/ge_events_df.csv`
- Utilise `RecursiveCharacterTextSplitter` pour découper le **document** *texte_rag* en segments:

    - taille des segments: `chunk_size = 700`
    - recouvrement: `chunk_overlap = 200`

- Génère une liste de chunks contenant:

    - métadonnées (uid, titre, ville, dates…)

    - texte du chunk

- Sauvegarde les chunks dans un fichier pickle: `data/chunks.pkl`.

### Embedding - Vectorisation des chunks
- Chargement des chunks
- Initialisation du client Mistral (`mistral-embed`)
- Vectorisation des chunks par batch de 64
- Gestion automatique des erreurs de rate limit
- Construit une structure contenant:

    - métadonnées

    - chunk de texte

    - embedding vectoriel

- Sauvegarde le tout dans `embeddings.pkl`.

### Indexation - Construction de l’index vectoriel **FAISS**
Objectif: Créer un index vectoriel **FAISS** à partir des embeddings générés précédemment, afin de permettre la recherche sémantique dans le chatbot.

- Charge les embeddings vectorisés: `data/embeddings.pkl`

- Convertit les embeddings en tableau NumPy float32 (format requis par FAISS)

- Initialise un index FAISS (IndexFlatL2) basé sur la distance euclidienne

- Ajoute tous les vecteurs dans l’index

- Sauvegarde l’index sur disque (`faiss.idx`)

- Sauvegarde les métadonnées associées à chaque vecteur (`metadata.pkl`): uid, titre, ville, dates, lien, chunk de texte, statut actif

Ces métadonnées permettent de reconstruire la réponse du chatbot après une recherche FAISS.