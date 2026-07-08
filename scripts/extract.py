import json
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path

# Initialisations
url = ("https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda/records")

limit = 100 # limite du nombre de requêtes par page

# Périodes à récupérer
periodes = [
    ("2025-01-01T00:00:00+00:00", "2025-04-01T00:00:00+00:00"),
    ("2025-04-01T00:00:00+00:00", "2025-07-01T00:00:00+00:00"),
    ("2025-07-01T00:00:00+00:00", "2025-10-01T00:00:00+00:00"),
    ("2025-10-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    ("2026-01-01T00:00:00+00:00", "2026-04-01T00:00:00+00:00"),
    ("2026-04-01T00:00:00+00:00", "2026-07-01T00:00:00+00:00"),
    ("2026-07-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00"),
    ("2026-10-01T00:00:00+00:00", "2027-01-01T00:00:00+00:00"),]

# Création du dossier data s'il n'existe pas
data_dir = Path("data")
data_dir.mkdir(parents = True, exist_ok = True)

fichier_sortie = data_dir / "grand_est_events.json"

# Extraction
events = []

for debut, fin in periodes:
    print(f"\n=== Extraction {debut[:10]} - {fin[:10]} ===")

    # Paramètres de filtres
    params = {
        "refine": [
            "location_region:Grand Est"
        ],
        "where": (
            f'firstdate_begin >= "{debut}" '
            f'AND firstdate_begin < "{fin}"'
        ),
        "limit": limit,
        "offset": 0}

    response = requests.get(url, params = params)
    response.raise_for_status()

    data = response.json()
    total_count = data["total_count"]
    print(f"{total_count} événements trouvés.")

    if total_count >= 10000:
        print("ATTENTION : cette période dépasse la limite de l'API.")

    periode_events = []
    for offset in tqdm(range(0, total_count, limit)):
        params["offset"] = offset
        response = requests.get(url, params=params)
        response.raise_for_status()

        page = response.json()["results"]
        periode_events.extend(page)
    print(f"-- {len(periode_events)} récupérés") 

    events.extend(periode_events)

# Suppression des doublons si existent
unique_events = {}
for e in events:
    unique_events[e["uid"]] = e
events = list(unique_events.values())
print(f"\nTotal après suppression des doublons: {len(events)}")

# Sauvegarde
with open(fichier_sortie, "w", encoding="utf-8") as f:
    json.dump(events, f, ensure_ascii=False, indent=4)

print(f"\n{len(events)} événements enregistrés dans {fichier_sortie}") 
