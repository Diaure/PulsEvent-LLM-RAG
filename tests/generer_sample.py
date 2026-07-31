import pandas as pd

# Charger ton CSV final
df = pd.read_csv("./data/ge_events_df.csv")

print("Colonnes disponibles :", df.columns.tolist())
print("Nombre total d'événements :", len(df))

# Sélectionner un sample cohérent
sample_df = df.sample(n=50, random_state=42)

# Sauvegarder le sample pour les tests unitaires
sample_df.to_csv("./tests/ge_events_sample.csv", index=False)

print("Sample créé :", len(sample_df), "événements")
