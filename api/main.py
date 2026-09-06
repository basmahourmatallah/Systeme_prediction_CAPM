from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Chargement du modèle et des fichiers associés (une seule fois, au démarrage de l'API)
model = joblib.load("../models/random_forest_capm.pkl")
label_encoder = joblib.load("../models/label_encoder_capm.pkl")
colonnes_modele = joblib.load("../models/colonnes_modele.pkl")

app = FastAPI(title="API de prédiction de gravité des intoxications - CAPM")


# Définition de la structure des données attendues en entrée
class CasIntoxication(BaseModel):
    age: float
    sexe: str
    milieu: str
    region: str
    circonstance: str
    sous_circonstance: str
    type_produit: str
    voie: str
    tranche_age: str


@app.get("/")
def accueil():
    return {"message": "API CAPM opérationnelle. Utilisez /predict pour obtenir une prédiction."}


@app.post("/predict")
def predire_gravite(cas: CasIntoxication):
    # Construction d'une ligne de données au format attendu par le modèle
    donnees_brutes = pd.DataFrame([{
        "Age Année": cas.age,
        "Sexe": cas.sexe,
        "Milieu": cas.milieu,
        "Région": cas.region,
        "Circonstance": cas.circonstance,
        "Sous circonstance": cas.sous_circonstance,
        "type de produit1": cas.type_produit,
        " Voie 1": cas.voie,
        "Tranche_age": cas.tranche_age,
    }])

    # One-hot encoding, comme à l'entraînement
    donnees_encodees = pd.get_dummies(donnees_brutes)

    # Aligner les colonnes exactement comme celles du modèle (colonnes manquantes = 0)
    donnees_finales = donnees_encodees.reindex(columns=colonnes_modele, fill_value=0)

    # Prédiction
    prediction = model.predict(donnees_finales)[0]
    probabilites = model.predict_proba(donnees_finales)[0]

    return {
        "grade_predit": prediction,
        "probabilites": dict(zip(model.classes_, probabilites.round(3).tolist()))
    }