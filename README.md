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

## Développement du chatbot intelligent (RAG)

### Fonctionnement général

Le chatbot intelligent Puls-Event repose sur une architecture RAG (Retrieval-Augmented Generation) permettant d'interroger la base d'évènements indexés de manière sémantique.

Lorsqu'un utilisateur pose une question, plusieurs étapes sont exécutées:

1. La question est vectorisée à l'aide du modèle d'embedding Mistral.
2. Une recherche sémantique est effectuée dans l'index FAISS afin de récupérer les chunks les plus pertinents.
3. Les chunks récupérés sont filtrés et regroupés par évènement afin d'éviter les doublons.
4. Les métadonnées associées aux évènements sont utilisées pour reconstruire le contexte.
5. Le modèle de langage génère une réponse adaptée à la requête de l'utilisateur.

Le chatbot est donc capable de:

- rechercher des évènements par thématique
- proposer des activités adaptées aux contraintes exprimées par l'utilisateur
- répondre à des questions générales sur les évènements disponibles
- fournir les informations utiles (dates, lieu, description, lien vers l'évènement).

### Recherche sémantique

Le moteur de recherche utilise:

- l'index vectoriel FAISS construit précédemment
- les embeddings générés avec le modèle `mistral-embed`
- les métadonnées sauvegardées dans `metadata.pkl`.

La recherche s'effectue sur les chunks les plus proches de la requête utilisateur afin de récupérer uniquement les informations pertinentes.

### Génération des réponses

Une fois les évènements pertinents récupérés:

- les informations sont injectées dans un prompt construit dynamiquement
- le LLM reçoit uniquement le contexte nécessaire
- le modèle génère une réponse naturelle.

Lorsque plusieurs évènements correspondent à la demande, le chatbot peut retourner une liste structurée contenant:

- le titre de l'évènement
- la ville
- la date
- le lieu
- le lien OpenAgenda associé.

### Reconstruction de l'index

Une méthode dédiée permet également de reconstruire automatiquement l'ensemble du pipeline RAG.

Cette reconstruction exécute successivement:

- le preprocessing
- le chunking
- la génération des embeddings
- la création de l'index FAISS.

Cette fonctionnalité facilite la mise à jour régulière des données lorsque de nouveaux évènements sont disponibles.

## Évaluation du chatbot

Afin d'évaluer les performances du système RAG, plusieurs jeux de questions ont été construits.

L'évaluation porte notamment sur:

- la pertinence des évènements proposés
- la qualité des réponses générées
- la capacité du chatbot à comprendre des formulations variées
- la robustesse face aux questions hors contexte.

Les tests ont été réalisés sur différentes catégories de requêtes:

- recherche d'évènements culturels
- recherche d'activités selon une ville ou une période
- recommandations personnalisées
- questions générales sur les évènements disponibles.

Les réponses générées ont été exportées dans un fichier CSV afin de pouvoir être comparées et analysées.

Les principaux indicateurs étudiés sont:

- **F**aithfulness**: cohérence entre la réponse et le contexte récupéré ;
- **Answer Relevancy**: pertinence de la réponse apportée ;
- **Context Precision**: qualité des documents retrouvés ;
- **Context Recall**: capacité à récupérer les informations attendues.

Ces métriques permettent d'évaluer la qualité globale du pipeline RAG et d'identifier les pistes d'amélioration du système.

## Création d'une API REST

Le chatbot est exposé sous la forme d'une API REST développée avec FastAPI.

L'API permet :

- d'interroger le chatbot via une requête HTTP ;
- de reconstruire automatiquement l'index vectoriel ;
- d'intégrer facilement le système RAG dans une application tierce.

L'API est lancée depuis la racine du projet: `uvicorn api.api_rag:app --reload`, et est accessible sur `
http://localhost:8000`. 

La documentation interactive Swagger est disponible sur `
http://localhost:8000/docs`

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


````markdown
## Interface utilisateur avec Streamlit

Une interface graphique a été développée avec Streamlit afin de rendre l'utilisation du chatbot plus intuitive.

L'application permet:

- de dialoguer avec le chatbot via une interface conversationnelle
- d'afficher les évènements sous forme de cartes interactives
- de conserver l'historique des échanges durant la session
- d'interroger directement l'API REST.

Les évènements proposés sont automatiquement affichés avec:

- leur titre
- leur ville
- leur date
- leur lieu
- un lien vers la page officielle OpenAgenda.

### Lancement de l'application

Après avoir lancé l'API FastAPI, lancer l'interface Streamlit:

```bash
streamlit scripts/run rag_streamlit.py