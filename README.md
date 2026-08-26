# Puls Event Intelligent Chatbot

## Extraction des données OpenAgenda

### Source des données

Les données proviennent du jeu de données OpenAgenda disponible sur OpenDataSoft :

`https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda`

API utilisée :

`https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records`


### Objectif

Afin d'alimenter le système RAG, les événements ont été extraits depuis le jeu de données **Événements publics Open Agenda** disponible sur l'API ***Open Agenda***.

L'extraction a été limitée:
- à la région **Grand Est**
- aux évènements compris entre le 1er janvier 2025 et le 31 décembre 2026

### Principes
L'API Open Agenda impose une limite: `offset + limit <= 10000`.

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
- Construit un champ textuel complet **texte_rag** utilisé pour le RAG
- Exporte les données nettoyées au format JSON + CSV: 

`data/ge_event_rag.json, data/ge_events_df.csv`.

### Chunking - Découpage des textes
Le script `chunk.py` quant à lui:

- Récupère le dataframe des évènements filtrés précédemment `data/ge_events_df.csv`
- Utilise `RecursiveCharacterTextSplitter` pour découper le **document - texte_rag** en segments:

    - taille des segments: `chunk_size = 700`
    - recouvrement: `chunk_overlap = 200`

- Génère une liste de chunks contenant:

    - métadonnées (uid, titre, ville, dates…)

    - texte du chunk

- Sauvegarde les chunks dans un fichier pickle: `data/chunks.pkl`.

### Embedding - Vectorisation des chunks
- Chargement des chunks
- Initialisation du client ***Mistral (`mistral-embed`)***
- Vectorisation des chunks par batch de 32
- Gestion automatique des erreurs de rate limit
- Construit une structure contenant:

    - métadonnées

    - chunk de texte

    - embedding vectoriel

- Sauvegarde le tout dans `embeddings.pkl`.

### Indexation - Construction de l’index vectoriel **FAISS**
Objectif: Créer un index vectoriel **FAISS** à partir des embeddings générés précédemment, afin de permettre la recherche sémantique dans le chatbot.

- Charge les embeddings vectorisés: `data/embeddings.pkl`

- Convertit les embeddings en tableau NumPy **`float32`** (format requis par FAISS)

- Initialise un index **`FAISS IndexFlatL2`** basé sur la distance euclidienne

- Ajoute tous les vecteurs dans l’index

- Sauvegarde l’index sur disque (`faiss.idx`)

- Sauvegarde les métadonnées associées à chaque vecteur (`metadata.pkl`): uid, titre, ville, dates, lien, chunk de texte, statut actif

Ces métadonnées permettent de reconstruire la réponse du chatbot après une recherche FAISS.

## Développement du chatbot intelligent (RAG)

Le chatbot intelligent Puls Event repose sur une architecture **RAG (Retrieval-Augmented Generation)** permettant d'interroger la base d'événements indexés de manière sémantique.

L'architecture du système peut être décrite selon deux niveaux complémentaires:

- **l'architecture globale du RAG**, qui présente l'ensemble du pipeline, depuis l'extraction des données jusqu'à la génération de la réponse.
- **le pipeline détaillé de génération de réponse**, qui précise les traitements effectués lorsqu'un utilisateur soumet une requête, notamment l'extraction des contraintes et l'application des filtres métier.

### Architecture globale

![architecture](https://raw.githubusercontent.com/Diaure/PulsEvent-LLM-RAG/main/Images/Image1.png)

Le pipeline global est composé de deux grandes phases.

- **Phase de préparation et d'indexation**: phase qui correspond au pipeline décrit dans la section `Indexation — Preprocessing, Chunking, Embedding et FAISS`.

- **Phase de recherche et de génération**
Lorsqu'un utilisateur pose une question, le système utilise l'index précédemment construit pour retrouver les événements pertinents et générer une réponse à partir des informations récupérées.

Cependant, pour Puls-Event, cette phase est enrichie par un pipeline spécifique permettant de prendre explicitement en compte les contraintes exprimées par l'utilisateur.

### Pipeline de génération de réponse

![pipeline](https://raw.githubusercontent.com/Diaure/PulsEvent-LLM-RAG/main/Images/Image2.png)

Le pipeline de génération de réponse constitue un niveau de détail supplémentaire de la phase **Recherche → Génération** du RAG.

Lorsqu'une question est reçue, le système commence par analyser la requête afin d'en extraire les contraintes utiles à la recherche d'événements.

**1. Extraction des contraintes**

La requête utilisateur est analysée afin d'identifier les différentes contraintes pouvant être utilisées pour sélectionner les événements.

Les principales contraintes prises en compte sont **Ville, Date / temporalité, Âge, Type d'événement**

Par exemple, pour une requête telle que:

*« Y a-t-il des concerts à Strasbourg ce week-end ? »*, le système peut identifier:

- Ville : Strasbourg
- Date / temporalité : ce week-end
- Âge : nonspécifié
- Type d'événement : concert.

Ces contraintes sont ensuite utilisées dans les étapes de recherche et de filtrage.

**2. Recherche des événements candidats**

La requête utilisateur est également utilisée pour effectuer une recherche sémantique dans l'index FAISS.

Le système recherche les événements dont les représentations vectorielles sont les plus proches de la requête.

Cette étape permet d'obtenir un ensemble *`d'événements candidats`* susceptibles de répondre à la demande.

La recherche sémantique constitue ainsi le mécanisme de *`retrieval du RAG`*.

**3. Application des filtres métier**

Les événements candidats sont ensuite confrontés aux contraintes extraites de la requête.

Des filtres métier peuvent notamment être appliqués sur:

- la ville ;
- la date ou la période ;
- l'âge ;
- le type d'événement.

Cette étape permet de compléter la recherche sémantique par des règles basées sur les métadonnées des événements.

Le système ne se limite donc pas à la proximité vectorielle: les contraintes explicites de l'utilisateur sont également prises en compte dans la sélection des résultats.

**4. Analyse et sélection des événements**

Après l'application des filtres métier, les événements restants sont analysés afin de sélectionner les éléments les plus pertinents pour répondre à la requête.

Cette étape permet de constituer le contexte final transmis au modèle de langage.

**5. Génération de la réponse**

Le LLM reçoit alors la question de l'utilisateur ainsi que les événements sélectionnés et leurs informations associées.

Il génère une réponse naturelle et contextualisée à partir de ces éléments.

Lorsque plusieurs événements correspondent à la demande, la réponse peut notamment présenter:

- le titre de l'événement ;
- la ville ;
- la date ;
- le lieu ;
- la description ;
- le lien OpenAgenda.

L'objectif est de produire une réponse basée sur les événements effectivement récupérés et sélectionnés par le pipeline.

Ainsi, les deux architectures sont complémentaires.

### Reconstruction de l'index

Une méthode dédiée permet également de reconstruire automatiquement l'ensemble du pipeline RAG.

Cette reconstruction exécute successivement:

- le preprocessing
- le chunking
- la génération des embeddings
- la création de l'index FAISS.

Cette fonctionnalité facilite la mise à jour régulière des données lorsque de nouveaux évènements sont disponibles.

## Évaluation du chatbot

L'évaluation porte sur les différentes étapes du pipeline RAG afin de mesurer à la fois la qualité de la récupération des événements et la qualité des réponses générées.

Plusieurs jeux de questions ont été construits afin de couvrir différents types de requêtes:

- recherche d'événements culturels
- recherche par ville
- recherche par période
- recherche par âge
- recherche par type d'événement
- recommandations combinant plusieurs contraintes
- formulations variées d'une même demande
- questions hors contexte.

Les performances du système sont notamment évaluées à l'aide des métriques suivantes:

- **Faithfulness**: cohérence entre la réponse et le contexte récupéré ;
- **Answer Relevancy**: pertinence de la réponse apportée ;
- **Context Precision**: qualité des documents retrouvés ;
- **Context Recall**: capacité à récupérer les informations attendues.

| Métrique | Score global |
|---|---:|
| Faithfulness | **0.69** |
| Answer Relevancy | **0.45** |
| Context Precision | **0.78** |
| Context Recall | **0.50** |


## Création d'une API REST

Le chatbot est exposé sous la forme d'une **API REST développée avec FastAPI**.

L'API permet :

- d'interroger le chatbot via une requête HTTP ;
- de reconstruire automatiquement l'index vectoriel ;
- d'intégrer facilement le système RAG dans une application tierce.

**Dockerisation**

L'API est également exécutée dans un conteneur **Docker**, afin de faciliter le déploiement et de garantir un environnement d'exécution reproductible.

Le conteneur permet notamment de regrouper les dépendances nécessaires au fonctionnement de l'API et du chatbot.

**Lancement de l'API**
- `Sans Docker`: depuis la racine du projet: 
`uvicorn api.api_rag:app --reload`, sur `http://localhost:8000`. 

La documentation interactive Swagger est disponible sur 
`http://localhost:8000/docs`

- `Avec Docker`:

L'image Docker peut être construite avec: `docker build -t pulsevent-rag .`

Puis créer le conteneur et le lancer avec: `docker run --pulsevent-rag-container -p 7860:7860 -e PULSEVENT-MISTRAL.KEY="%PULSEVENT_MISTRAL_KEY% pulsevent-rag:latest`.

L'API est alors accessible sur `http://localhost:7860/docs`.

### Endpoints disponibles

1. **Poser une question**

Endpoint: ***POST/ask***

`json
{"question": "Quels sont les évènements organisés à Strasbourg ce week-end ?"}`

***Réponse***

`json
{
    "question": "...",
    "answer": "..."
}`

2. **Reconstruire l'index**

Endpoint: ***POST/rebuild***

`json
{"statut": "Index reconstruit avec succès"}`
