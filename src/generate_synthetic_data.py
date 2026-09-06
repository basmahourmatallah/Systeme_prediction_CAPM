"""
Génération d'un dataset synthétique réaliste pour le projet
"Prédiction de la gravité des intoxications - CAPM"

Ce script simule des données EN ATTENDANT la vraie base du CAPM.
Les relations entre variables sont construites pour ressembler
à ce que montre la littérature (âge, agent toxique, délai de PEC
influencent la gravité), mais restent une SIMULATION.

Usage :
    python generate_synthetic_data.py
Génère : data/raw/donnees_synthetiques_capm.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Reproductibilité
np.random.seed(42)

N = 5200  # nombre de cas

# ------------------------------------------------------------------
# 1. Variables démographiques
# ------------------------------------------------------------------
age = np.clip(np.random.gamma(shape=2.2, scale=12, size=N), 0, 95).astype(int)
sexe = np.random.choice(["F", "M"], size=N, p=[0.55, 0.45])

regions = [
    "Rabat-Salé-Kénitra", "Casablanca-Settat", "Marrakech-Safi",
    "Fès-Meknès", "Tanger-Tétouan-Al Hoceïma", "Oriental",
    "Souss-Massa", "Béni Mellal-Khénifra", "Drâa-Tafilalet",
    "Guelmim-Oued Noun", "Laâyoune-Sakia El Hamra", "Eddakhla-Oued Eddahab"
]
region_probs = [0.18, 0.22, 0.12, 0.11, 0.09, 0.07, 0.07, 0.05, 0.04, 0.02, 0.02, 0.01]
region = np.random.choice(regions, size=N, p=region_probs)

milieu = np.where(
    np.random.random(N) < 0.78, "urbain", "rural"
)  # cohérent avec la littérature CAPM (~90% urbain sur certaines études, on varie un peu)

# ------------------------------------------------------------------
# 2. Circonstances
# ------------------------------------------------------------------
def circonstance_from_age(a):
    if a < 15:
        return np.random.choice(
            ["accidentelle", "volontaire", "criminelle"], p=[0.92, 0.03, 0.05]
        )
    elif a < 25:
        return np.random.choice(
            ["accidentelle", "volontaire", "criminelle"], p=[0.45, 0.50, 0.05]
        )
    else:
        return np.random.choice(
            ["accidentelle", "volontaire", "criminelle", "professionnelle"],
            p=[0.40, 0.35, 0.05, 0.20]
        )

circonstance = np.array([circonstance_from_age(a) for a in age])

agents = [
    "medicament", "pesticide", "produit_menager", "monoxyde_carbone",
    "drogue", "cosmetique", "plante_venin", "alimentaire"
]

def agent_from_age_circ(a, c):
    if a < 6:
        return np.random.choice(
            agents, p=[0.30, 0.05, 0.45, 0.05, 0.01, 0.09, 0.03, 0.02]
        )
    if c == "volontaire":
        return np.random.choice(
            agents, p=[0.55, 0.20, 0.10, 0.05, 0.05, 0.02, 0.01, 0.02]
        )
    if c == "professionnelle":
        return np.random.choice(
            agents, p=[0.05, 0.35, 0.20, 0.30, 0.01, 0.02, 0.02, 0.05]
        )
    return np.random.choice(
        agents, p=[0.25, 0.12, 0.20, 0.15, 0.08, 0.08, 0.05, 0.07]
    )

agent_toxique = np.array([agent_from_age_circ(a, c) for a, c in zip(age, circonstance)])

voie_exposition = np.array([
    "orale" if ag in ["medicament", "drogue", "alimentaire", "plante_venin"]
    else np.random.choice(["inhalation", "orale"], p=[0.85, 0.15]) if ag == "monoxyde_carbone"
    else np.random.choice(["cutanee", "orale", "inhalation"], p=[0.5, 0.3, 0.2])
    for ag in agent_toxique
])

co_exposition = np.random.random(N) < 0.12  # polyintoxication

# ------------------------------------------------------------------
# 3. Délai et prise en charge (disponibles au moment de l'appel)
# ------------------------------------------------------------------
delai_exposition_appel_h = np.round(np.clip(np.random.exponential(scale=2.5, size=N), 0.1, 48), 1)
premiers_secours_effectues = np.random.random(N) < 0.35

# ------------------------------------------------------------------
# 4. Symptômes initiaux (dépendent de l'agent + quantité latente)
# ------------------------------------------------------------------
quantite_latente = np.clip(np.random.gamma(shape=1.8, scale=1.0, size=N), 0.1, 8)

agent_toxicite_base = {
    "medicament": 1.0, "pesticide": 2.2, "produit_menager": 0.7,
    "monoxyde_carbone": 1.8, "drogue": 1.3, "cosmetique": 0.3,
    "plante_venin": 1.1, "alimentaire": 0.5
}
toxicite_agent = np.array([agent_toxicite_base[a] for a in agent_toxique])

# Score de gravité latent (continu), influencé par plusieurs facteurs réalistes
score_latent = (
    0.4 * toxicite_agent
    + 0.35 * quantite_latente
    + 0.02 * np.maximum(0, age - 60) / 10          # sujets âgés plus vulnérables
    + 0.015 * np.maximum(0, 5 - age)               # très jeunes enfants plus vulnérables
    + 0.25 * co_exposition.astype(int)
    + 0.15 * np.clip(delai_exposition_appel_h, 0, 12) / 12
    - 0.3 * premiers_secours_effectues.astype(int)
    + np.random.normal(0, 0.6, N)                  # bruit
)

symptomes_neuro = (score_latent + np.random.normal(0, 0.5, N)) > 1.8
symptomes_cardioresp = (score_latent + np.random.normal(0, 0.5, N)) > 2.0
symptomes_digestifs = np.random.random(N) < np.clip(0.3 + 0.1 * toxicite_agent, 0, 0.9)

score_glasgow = np.clip(15 - np.round(np.clip(score_latent, 0, 6) * 1.8), 3, 15).astype(int)
freq_cardiaque = np.clip(80 + score_latent * 12 + np.random.normal(0, 10, N), 40, 180).astype(int)
saturation_o2 = np.clip(98 - score_latent * 3 + np.random.normal(0, 2, N), 60, 100).astype(int)

# ------------------------------------------------------------------
# 5. Variable cible : PSS (0 à 4), dérivée du score latent
# ------------------------------------------------------------------
def latent_to_pss(s):
    if s < 0.5:
        return 0
    elif s < 1.5:
        return 1
    elif s < 2.8:
        return 2
    elif s < 4.2:
        return 3
    else:
        return 4

pss = np.array([latent_to_pss(s) for s in score_latent])

# petite proportion de décès parmi les PSS=4 les plus extrêmes (cohérence logique)
issue_finale = np.where(
    pss == 4, "deces",
    np.where(pss == 3, "hospitalisation_reanimation",
             np.where(pss == 2, "hospitalisation_surveillance",
                      np.where(pss == 1, "ambulatoire", "sans_suite")))
)

# ------------------------------------------------------------------
# 6. Dates
# ------------------------------------------------------------------
start_date = datetime(2019, 1, 1)
dates = [start_date + timedelta(days=int(d)) for d in np.random.uniform(0, 5 * 365, N)]

# ------------------------------------------------------------------
# 7. Assemblage du DataFrame
# ------------------------------------------------------------------
df = pd.DataFrame({
    "id_cas": [f"CAPM-{i:05d}" for i in range(1, N + 1)],
    "date_declaration": dates,
    "region": region,
    "milieu": milieu,
    "age": age,
    "sexe": sexe,
    "circonstance": circonstance,
    "agent_toxique": agent_toxique,
    "voie_exposition": voie_exposition,
    "co_exposition": co_exposition,
    "delai_exposition_appel_h": delai_exposition_appel_h,
    "premiers_secours_effectues": premiers_secours_effectues,
    "symptomes_neurologiques": symptomes_neuro,
    "symptomes_cardiorespiratoires": symptomes_cardioresp,
    "symptomes_digestifs": symptomes_digestifs,
    "score_glasgow": score_glasgow,
    "frequence_cardiaque": freq_cardiaque,
    "saturation_o2": saturation_o2,
    "PSS": pss,
    "issue_finale": issue_finale,
})

# ------------------------------------------------------------------
# 8. Introduire des valeurs manquantes réalistes (les vraies données CAPM en ont)
# ------------------------------------------------------------------
for col, taux_manquant in [
    ("score_glasgow", 0.15),
    ("saturation_o2", 0.20),
    ("frequence_cardiaque", 0.10),
    ("milieu", 0.05),
]:
    mask = np.random.random(N) < taux_manquant
    df.loc[mask, col] = np.nan

# Sauvegarde
import os
os.makedirs("data/raw", exist_ok=True)
output_path = "data/raw/donnees_synthetiques_capm.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Dataset généré : {output_path}")
print(f"Nombre de cas : {len(df)}")
print(f"\nDistribution du PSS (variable cible) :")
print(df["PSS"].value_counts().sort_index())
print(f"\nAperçu :")
print(df.head())
