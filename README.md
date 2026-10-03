# Système d'aide à la décision — Prédiction de la gravité des intoxications (CAPM)

Projet de fin d'année (PFA) — INSEA.

🔗 **Application** : [intoxicationprediction.streamlit.app](https://intoxicationprediction.streamlit.app/)
📄 **Rapport complet** : [`rapport/Rapport_PFA.pdf`](./rapport/Rapport_PFA.pdf)

## Objectif

Prédire automatiquement la gravité d'une intoxication à partir des données du
Centre Anti Poison et de Pharmacovigilance du Maroc (CAPM), pour appuyer la
décision clinique du personnel médical.

## Structure du projet

- `notebooks/` : nettoyage, EDA, modélisation, SHAP
- `dashboard/` : application Streamlit (interface de prédiction)
- `models/` : modèle final entraîné et artefacts associés
- `src/` : fonctions réutilisables (prétraitement, entraînement)
- `data/` : données brutes et traitées (non versionnées, confidentielles)

## Modèle final

Random Forest optimisé par `RandomizedSearchCV`, rééquilibrage **SMOTENC**
intégré à la validation croisée (respecte la nature catégorielle des
variables), variable `Produit_Regroupe`, fusion des grades 3-4 en classe
"Sévère". F1-macro : 0,513 — Rappel Sévère : 46,1 % (seuil optimisé à 0,25).

## Lancer le projet

```bash
pip install -r requirements.txt
cd dashboard
streamlit run app.py
```
