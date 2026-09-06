# Système d'aide à la décision — Prédiction de la gravité des intoxications (CAPM)

Projet de fin d'année (PFA) — INSEA, filière Biostatistique, Démographie et Big Data.

## Objectif
Prédire automatiquement la gravité d'une intoxication à partir des données du Centre Anti Poison
et de Pharmacovigilance du Maroc (CAPM), pour appuyer la décision clinique du personnel médical.

## Structure du projet
- `notebooks/` : nettoyage, EDA, modélisation, SHAP
- `dashboard/` : application Streamlit (interface de prédiction)
- `models/` : modèle final entraîné et artefacts associés
- `data/` : données brutes et traitées (non versionnées si volumineuses)

## Modèle final
Random Forest optimisé, pipeline SMOTE intégré à la validation croisée,
variable `Produit_Regroupe`, fusion des grades 3-4 en classe "Sévère".

## Lancer le projet
\`\`\`bash
pip install -r requirements.txt
cd dashboard
streamlit run app.py
\`\`\`
