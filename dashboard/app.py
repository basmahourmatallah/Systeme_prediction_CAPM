import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import re
import unicodedata
import base64
import os

st.set_page_config(page_title="CAPM — Système d'aide à la décision", page_icon="🏥", layout="wide")

# ─── THEME ───
st.markdown("""
<style>
:root {
    --bg-main: #0e1420; --bg-panel: #161d2b; --bg-panel-hover: #1c2536;
    --accent: #3ea6ff; --accent-dark: #1a5276; --text-main: #e8ecf1;
    --text-muted: #8b96a8; --border: #232d40;
}
.stApp { background-color: var(--bg-main); color: var(--text-main); }
.capm-header { text-align: center; padding: 28px 16px 20px 16px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
.capm-header h1 { color: var(--text-main); font-size: 26px; font-weight: 700; margin-bottom: 4px; }
.capm-header p { color: var(--accent); font-size: 15px; font-weight: 500; margin: 0; }
.capm-card { background-color: var(--bg-panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px 22px; margin-bottom: 16px; }
.capm-card h4 { color: var(--accent); margin-top: 0; font-size: 15px; text-transform: uppercase; }
.kpi-card { background: linear-gradient(145deg, #161d2b 0%, #131a29 100%); border: 1px solid var(--border); border-radius: 14px; padding: 20px 18px; text-align: center; }
.kpi-value { font-size: 26px; font-weight: 800; color: var(--accent); margin: 2px 0; }
.kpi-label { color: var(--text-muted); font-size: 12.5px; text-transform: uppercase; }
.glow-card { background: linear-gradient(160deg, #161d2b 0%, #131a29 100%); border: 1px solid var(--border); border-radius: 16px; padding: 26px; height: 100%; }
.glow-card h4 { color: var(--text-main); font-size: 16px; font-weight: 700; margin: 0 0 10px 0; }
.glow-card p { color: var(--text-muted); font-size: 14.5px; line-height: 1.65; }
.hero { text-align: center; padding: 48px 24px 44px 24px; border-radius: 18px; margin-bottom: 28px;
        background: linear-gradient(180deg, #131a29 0%, #0e1420 100%); border: 1px solid var(--border); }
.hero-title { font-size: 34px; font-weight: 800; margin: 0 0 12px 0; color: white; }
.hero-sub { color: var(--text-muted); font-size: 16px; max-width: 640px; margin: 0 auto; }
.capm-footer { text-align: center; color: var(--text-muted); font-size: 13px; padding-top: 24px; border-top: 1px solid var(--border); margin-top: 32px; }
</style>
""", unsafe_allow_html=True)

# ─── FONCTIONS DE NETTOYAGE (identiques au notebook) ───
def retirer_accents(val):
    return ''.join(c for c in unicodedata.normalize('NFD', val) if unicodedata.category(c) != 'Mn')

def nettoyer_produit(val):
    val = str(val).strip().upper()
    val = re.sub(r'\s+', ' ', val)
    return retirer_accents(val)

# ─── CHARGEMENT DES ARTEFACTS ───
import pathlib

# Chemin absolu du dossier où se trouve ce script (dashboard/), peu importe d'où il est lancé
DOSSIER_APP = pathlib.Path(__file__).parent
DOSSIER_MODELS = DOSSIER_APP.parent / "models"

@st.cache_resource
def charger_artefacts():
    pipeline = joblib.load(DOSSIER_MODELS / "pipeline_randomforest_smotenc.pkl")
    colonnes_features = joblib.load(DOSSIER_MODELS / "colonnes_features.pkl")
    produits_frequents = joblib.load(DOSSIER_MODELS / "liste_produits_frequents.pkl")
    valeurs_categorielles = joblib.load(DOSSIER_MODELS / "valeurs_categorielles.pkl")
    synonymes = joblib.load(DOSSIER_MODELS / "synonymes_produits.pkl")
    seuil_optimal = joblib.load(DOSSIER_MODELS / "seuil_optimal_severe.pkl")
    return pipeline, colonnes_features, produits_frequents, valeurs_categorielles, synonymes, seuil_optimal

try:
    pipeline, colonnes_features, produits_frequents, valeurs_categorielles, synonymes, seuil_optimal = charger_artefacts()
except Exception as e:
    st.error("Erreur lors du chargement des artefacts du modèle :")
    st.exception(e)
    st.stop()

def regrouper_produit(val):
    val = nettoyer_produit(val)
    val = synonymes.get(val, val)
    return val if val in produits_frequents else "AUTRE"

if "historique" not in st.session_state:
    st.session_state.historique = []

# ─── EN-TÊTE AVEC LOGO ───
def _logo_base64(path=None):
    if path is None:
        path = DOSSIER_APP / "assets" / "logo_capm.jpg"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

_logo_b64 = _logo_base64()
_logo_html = f'<img src="data:image/jpeg;base64,{_logo_b64}" style="height:64px;margin-bottom:12px;">' if _logo_b64 else ""

st.markdown(f"""
    <div class="capm-header">
        {_logo_html}
        <h1>Centre Anti Poison et de Pharmacovigilance du Maroc</h1>
        <p>Système d'aide à la décision — Prédiction de la gravité des intoxications</p>
    </div>
""", unsafe_allow_html=True)

tab_accueil, tab_prediction, tab_historique = st.tabs(["Accueil", "Prédiction", "Historique"])

# ══════════════════ ACCUEIL ══════════════════
with tab_accueil:
    st.markdown("""
        <div class="hero">
            <div class="hero-title">Une évaluation clinique instantanée, fondée sur des milliers de cas réels</div>
            <p class="hero-sub">Modèle Random Forest avec SMOTENC, entraîné sur les données du CAPM pour orienter la prise en charge.</p>
        </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    for col, value, label in [(k1,"22 188","Cas analysés"), (k2,"2020–2025","Période couverte"),
                                (k3,"4","Classes de gravité"), (k4,"RF + SMOTENC","Modèle final")]:
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""<div class="glow-card"><h4>Objectif</h4>
            <p>Prédire automatiquement la <b>gravité d'une intoxication</b> pour appuyer la décision clinique du personnel du CAPM.</p></div>""", unsafe_allow_html=True)
    with col_b:
        texte_seuil = f"Seuil de décision optimal calibré à {seuil_optimal:.2f} (maximise le F2-score sur la classe Sévère)."
        html_modele = (
            '<div class="glow-card"><h4>Modèle utilisé</h4>'
            '<p>Pipeline <b>SMOTENC + Random Forest</b> : le rééquilibrage respecte la nature catégorielle des variables '
            '(contrairement à SMOTE classique qui interpole des colonnes one-hot). Grades 3 et 4 fusionnés en <b>« Sévère »</b>. '
            + texte_seuil +
            '</p></div>'
        )
        st.markdown(html_modele, unsafe_allow_html=True)

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    data_pss = {
        "Classe": ["Grade 0", "Grade 1", "Grade 2", "Sévère (3-4)"],
        "Description": ["Aucun signe clinique", "Signes légers spontanément régressifs",
                         "Signes prononcés nécessitant un traitement", "Intoxication sévère avec risque vital, ou décès"],
        "Conduite à tenir": ["Surveillance à domicile", "Consultation médicale", "Hospitalisation", "Réanimation urgente"],
    }
    st.table(pd.DataFrame(data_pss))

# ══════════════════ PRÉDICTION ══════════════════
with tab_prediction:
    st.markdown('<div class="capm-card"><h4>Informations du cas</h4>', unsafe_allow_html=True)
    col_g, col_d = st.columns(2)

    with col_g:
        age = st.number_input("Âge (années)", min_value=0, max_value=110, value=25)
        sexe = st.selectbox("Sexe", valeurs_categorielles["Sexe"])
        milieu = st.selectbox("Milieu", valeurs_categorielles# ─── CHARGEMENT DES ARTEFACTS (chemins corrigés avec ../) ───["Milieu"])
        region = st.selectbox("Région", valeurs_categorielles["Région"])
        circonstance = st.selectbox("Circonstance", valeurs_categorielles["Circonstance"])

    with col_d:
        sous_circonstance = st.selectbox("Sous circonstance", valeurs_categorielles["Sous circonstance"])
        type_produit = st.selectbox("Type de produit", valeurs_categorielles["type de produit1"])
        produit_brut = st.text_input("Nom du produit (tel que rapporté)", value="Inconnu")
        voie = st.selectbox("Voie d'exposition", valeurs_categorielles[" Voie 1"])
        tranche_age = st.selectbox("Tranche d'âge", valeurs_categorielles["Tranche_age"])

    st.markdown('</div>', unsafe_allow_html=True)

    _, col_btn, _ = st.columns([1,1,1])
    with col_btn:
        predict_btn = st.button("🔍 Prédire la gravité", use_container_width=True)

    if predict_btn:
        produit_regroupe = regrouper_produit(produit_brut)

        ligne = pd.DataFrame([{
            "Age Année": age, "Sexe": sexe, "Milieu": milieu, "Région": region,
            "Circonstance": circonstance, "Sous circonstance": sous_circonstance,
            "type de produit1": type_produit, " Voie 1": voie,
            "Tranche_age": tranche_age, "Produit_Regroupe": produit_regroupe,
        }])[colonnes_features]

        prediction = pipeline.predict(ligne)[0]
        probabilites = pipeline.predict_proba(ligne)[0]
        prob_max = probabilites.max() * 100
        classes = pipeline.named_steps['rf'].classes_

        st.markdown("---")
        st.markdown('<div class="capm-card"><h4>Résultat de la prédiction</h4>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if prediction == "Grade 0":
                st.success(f"✅ {prediction} — Aucun signe clinique")
                st.info("Surveillance à domicile recommandée")
            elif prediction == "Grade 1":
                st.info(f"🔵 {prediction} — Signes légers")
                st.info("Consultation médicale recommandée")
            elif prediction == "Grade 2":
                st.warning(f"⚠️ {prediction} — Signes prononcés")
                st.warning("Hospitalisation recommandée")
            else:
                st.error(f"🚨 {prediction} — Intoxication sévère ou décès")
                st.error("Prise en charge urgente / réanimation")

            st.metric("Probabilité de la classe prédite", f"{prob_max:.1f}%")

            if prediction != "Sévère (3-4)":
                proba_severe = probabilites[list(classes).index("Sévère (3-4)")] * 100
                if proba_severe >= seuil_optimal * 100:
                    st.caption(f"⚠️ Le modèle attribue {proba_severe:.1f}% de probabilité à la classe "
                               f"'Sévère' (seuil calibré : {seuil_optimal*100:.0f}%). "
                               "Une prédiction non sévère ne doit pas remplacer le jugement clinique.")

        with col2:
            fig = go.Figure(go.Bar(
                x=[str(c) for c in classes],
                y=[p*100 for p in probabilites],
                marker_color=["#2ecc71","#3498db","#f39c12","#e74c3c"][:len(classes)]
            ))
            fig.update_layout(title="Probabilité par classe", yaxis_title="Probabilité (%)",
                               height=300, paper_bgcolor="#161d2b", plot_bgcolor="#161d2b", font_color="#e8ecf1")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.session_state.historique.append({
            "Âge": age, "Sexe": sexe, "Région": region, "Circonstance": circonstance,
            "Produit": produit_brut, "Produit regroupé": produit_regroupe,
            "Résultat": prediction, "Probabilité": f"{prob_max:.1f}%",
        })

# ══════════════════ HISTORIQUE ══════════════════
with tab_historique:
    st.markdown('<div class="capm-card"><h4>Historique des prédictions de cette session</h4>', unsafe_allow_html=True)
    if len(st.session_state.historique) == 0:
        st.info("Aucune prédiction effectuée pour le moment.")
    else:
        st.table(pd.DataFrame(st.session_state.historique))
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.historique = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div class='capm-footer'>© CAPM — Centre Anti Poison et de Pharmacovigilance du Maroc</div>", unsafe_allow_html=True)