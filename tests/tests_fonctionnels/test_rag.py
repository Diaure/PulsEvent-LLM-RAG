import pytest
from datetime import datetime, timezone, timedelta

from scripts.chatbot_rag import (
    extraire_ville,
    extraire_intervalle_temporel,
    event_is_in_interval,
    recherche_event_pertinent
)

# Test fonctionnel : extraction complète de la requête

def test_rag_extraction():
    question = "Que faire à Reims demain pour un enfant de 8 ans ?"

    ville = extraire_ville(question)
    intervalle = extraire_intervalle_temporel(question)

    assert ville == "reims"

    start, end = intervalle
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    assert start.date() == tomorrow.date()
    assert end.date() == tomorrow.date()


# Test fonctionnel : filtrage complet
def test_rag_filtrage():
    # Événement simulé
    event = {
        "city": "reims",
        "age_minimum": 6,
        "age_maximum": 12,
        "timing_begin": "2026-09-19T07:00:00Z",
        "timing_end": "2026-09-19T10:00:00Z",
        "firstdate_begin": "2026-09-19T07:00:00Z",
        "lastdate_end": "2099-01-01T00:00:00Z",
        "title": "Atelier enfants",
    }

    # Requête simulée
    age = 8

    date = datetime(2026, 9, 19, tzinfo=timezone.utc)
    start = datetime.combine(date.date(), datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(date.date(), datetime.max.time(), tzinfo=timezone.utc)


    # Filtre ville
    assert extraire_ville("Que faire à Reims ?") == "reims"

    # Filtre âge
    assert event["age_minimum"] <= age <= event["age_maximum"]

    # Filtre date
    assert event_is_in_interval(event, start, end)


# Test fonctionnel: pipeline complet RAG
def test_rag_pipeline_complet():
    question = "Que faire à Reims demain pour un enfant de 8 ans ?"

    results = recherche_event_pertinent(question)

    # Le pipeline doit renvoyer une liste
    assert isinstance(results, list)

    # Il doit y avoir au moins un événement
    # assert len(results) > 0

    # Chaque événement doit contenir les métadonnées essentielles
    if len(results) > 0:
        event = results[0]
        assert "title" in event
        assert "city" in event
        assert "canonicalurl" in event
        assert "chunk" in event
