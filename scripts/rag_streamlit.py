import streamlit as st
import json
import requests

# Fonction pour appeler l'API via streamlit
def appel_api_rest(question: str) -> str:
    url = "http://localhost:8000/ask"  # adapte si besoin
    payload = {"question": question}
    r = requests.post(url, json=payload)
    data = r.json()
    return data["answer"]

st.set_page_config(page_title="Assistant Virtuel Puls-Events", page_icon="🏛️")

# Initialisation de la session
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- STYLE CSS ---
st.markdown("""
<style>

.chat-bubble-user {
    background-color: #1E3A8A; /* bleu foncé */
    color: white;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 80%;
}

.chat-bubble-assistant {
    background-color: #2D2D2D; /* gris anthracite */
    color: white;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 80%;
}

.event-card {
    background-color: #1A1A1A; /* gris très foncé */
    border: 1px solid #444;
    padding: 12px 16px;
    border-radius: 10px;
    margin: 10px 0;
    color: white;
}

.event-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 6px;
    color: #4EA8DE; /* bleu clair */
}

.event-meta {
    font-size: 0.9rem;
    color: #CCCCCC;
}

a {
    color: #4EA8DE !important; /* bleu clair */
    text-decoration: none;
}

</style>
""", unsafe_allow_html=True)


# --- FONCTION D'AFFICHAGE DES ÉVÉNEMENTS ---
def afficher_evenements(response_text):
    """
    Si la réponse contient une liste d'événements en JSON,
    on les affiche sous forme de cartes.
    Sinon, on affiche le texte brut.
    """

    try:
        data = json.loads(response_text)

        if isinstance(data, list):
            # On a une liste d'événements
            for ev in data:
                st.markdown(f"""
                <div class="event-card">
                    <div class="event-title">{ev.get('title', 'Événement')}</div>
                    <div class="event-meta">📍 {ev.get('city', 'Ville inconnue')}</div>
                    <div class="event-meta">📅 {ev.get('date', 'Date inconnue')}</div>
                    <div class="event-meta">🏛️ {ev.get('lieu', 'Lieu inconnu')}</div>
                    <div class="event-meta">🔗 <a href="{ev.get('canonicalurl', '#')}" target="_blank">Plus d'informations</a></div>
                </div>
                """, unsafe_allow_html=True)
            return True

        return False

    except:
        return False


# --- AFFICHAGE DES MESSAGES ---
st.title("🏛️ Assistant Virtuel PULS-EVENTS")

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    if role == "user":
        st.markdown(f'<div class="chat-bubble-user">{content}</div>', unsafe_allow_html=True)
    else:
        # Assistant
        if not afficher_evenements(content):
            st.markdown(f'<div class="chat-bubble-assistant">{content}</div>', unsafe_allow_html=True)


# --- INPUT UTILISATEUR ---
if prompt := st.chat_input("Comment puis-je vous aider ?"):
    # Affichage du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-bubble-user">{prompt}</div>', unsafe_allow_html=True)
    print("PROMPT REÇU :", prompt)

    # Affichage du chargement
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown('<div class="chat-bubble-assistant">En train de réfléchir...</div>', unsafe_allow_html=True)

        # Appel à l'API
        try:
            response = appel_api_rest(prompt)
            print("Réponse API :", response)
        except Exception as e:
            print("ERREUR API :", e)

        # Affichage final
        if afficher_evenements(response):
            placeholder.empty()
        else:
            placeholder.markdown(f'<div class="chat-bubble-assistant">{response}</div>', unsafe_allow_html=True)

    # Ajout à l'historique
    st.session_state.messages.append({"role": "assistant", "content": response})
