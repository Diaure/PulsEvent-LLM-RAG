import pickle

fake_metadata = [
    {
        "uid": "1",
        "title": "Atelier famille",
        "city": "Mulhouse",
        "date": "2026-06-27T14:00:00Z",
        "lastdate_end": "2026-09-12T00:00:00Z",
        "conditions": "Gratuit",
        "age_min": 3,
        "age_max": 12,
        "canonicalurl": "https://example.com/atelier",
        "chunk": "Atelier pour enfants et parents."
    },
    {
        "uid": "2",
        "title": "Concert classique",
        "city": "Strasbourg",
        "date": "2026-08-15T20:00:00Z",
        "lastdate_end": "2026-08-15T22:00:00Z",
        "conditions": "Payant",
        "age_min": None,
        "age_max": None,
        "canonicalurl": "https://example.com/concert",
        "chunk": "Concert de musique classique."
    }
]

with open("fake_metadata.pkl", "wb") as f:
    pickle.dump(fake_metadata, f)

print("fake_metadata.pkl créé.")
