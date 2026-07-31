from scripts.chatbot_rag import event_is_in_interval, extraire_ville
from datetime import datetime, timezone


# Les filtres qui ne sont pas des fonctions sont testés en ré-exécutant leur logique inline dans le fichier de tests.
# Cela me permet de tester le comportement sans modifier le code de production.
# Je teste que la logique qui détermine si un événement est encore actif fonctionne correctement dans tous les cas : futur, passé, date absente ou invalide


# Test filtre si l'évènement est toujours d'actualité
def compute_event_actif_inline(event, today):
    last_date = event.get("lastdate_end")
    if last_date:
        try:
            date_fin = datetime.fromisoformat(last_date)
            return date_fin >= today
        except Exception:
            return False
    else:
        return False

def test_event_actif_future(): # Vérifie qu’un événement dont la date de fin est dans le futur est bien considéré comme actif
    today = datetime(2026, 7, 31, tzinfo=timezone.utc)
    event = {"lastdate_end": "2099-01-01T00:00:00"}
    assert compute_event_actif_inline(event, today) is True

def test_event_actif_past(): # Vérifie qu’un événement déjà terminé est bien marqué comme inactif
    today = datetime(2026, 7, 31, tzinfo=timezone.utc)
    event = {"lastdate_end": "2000-01-01T00:00:00"}
    assert compute_event_actif_inline(event, today) is False

def test_event_actif_missing(): # Vérifie que si la date de fin n’existe pas, l’événement est considéré comme inactif.
    today = datetime(2026, 7, 31, tzinfo=timezone.utc)
    event = {}
    assert compute_event_actif_inline(event, today) is False

def test_event_actif_invalid(): # Vérifie que si la date est invalide, l’événement est considéré comme inactif
    today = datetime(2026, 7, 31, tzinfo=timezone.utc)
    event = {"lastdate_end": "date_invalide"}
    assert compute_event_actif_inline(event, today) is False

# Test du filtre sur l'intervalle temporel, Il vérifie si l’événement se déroule dans l’intervalle demandé par l’utilisateur
# Je teste que l’extraction automatique de la ville fonctionne : elle détecte les villes présentes dans la question et renvoie None quand il n’y en a pas.
def test_interval_normal_true(): # Vérifie qu’un événement qui tombe exactement dans l’intervalle est accepté.
    event = {
        "timing_begin": "2026-09-19T07:00:00Z",
        "timing_end": "2026-09-19T10:00:00Z"
    }
    start = datetime(2026, 9, 19, tzinfo=timezone.utc)
    end = datetime(2026, 9, 19, tzinfo=timezone.utc)
    assert event_is_in_interval(event, start, end) is True

def test_interval_normal_false(): # Vérifie qu’un événement complètement en dehors de l’intervalle est rejeté
    event = {
        "timing_begin": "2026-09-19T07:00:00Z",
        "timing_end": "2026-09-19T10:00:00Z"
    }
    start = datetime(2026, 7, 6, tzinfo=timezone.utc)
    end = datetime(2026, 7, 6, tzinfo=timezone.utc)
    assert event_is_in_interval(event, start, end) is False

def test_interval_fallback_true():
    event = {
        "timing_begin": None,
        "timing_end": None,
        "firstdate_begin": "2026-09-19T07:00:00Z",
        "lastdate_end": "2026-09-19T10:00:00Z"
    }
    start = datetime(2026, 9, 19, tzinfo=timezone.utc)
    end = datetime(2026, 9, 19, tzinfo=timezone.utc)
    assert event_is_in_interval(event, start, end) is True

def test_interval_invalid(): #  Vérifie que si les dates sont invalides, la fonction renvoie False
    event = {
        "timing_begin": "date_invalide",
        "timing_end": "date_invalide"
    }
    start = datetime(2026, 9, 19, tzinfo=timezone.utc)
    end = datetime(2026, 9, 19, tzinfo=timezone.utc)
    assert event_is_in_interval(event, start, end) is False

# Test extraction des villes; normalise la question, normalise les villes de la base, cherche une ville dans la question, renvoie la ville trouvée ou None
def test_extract_city_reims(): # Vérifie que la ville 'Reims' est correctement détectée dans une question
    assert extraire_ville("Que faire à Reims demain") == "reims"

def test_extract_city_nancy(): # Vérifie que la ville 'Nancy' est correctement détectée
    assert extraire_ville("Événements à Nancy") == "nancy"

def test_extract_city_none(): # Vérifie qu’une question sans ville renvoie None
    assert extraire_ville("Que faire ce week-end ?") is None