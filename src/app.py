"""Fixed Streamlit entry point for the project template."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import MODEL_METRICS_FILE, DATA_DIR


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_DIR / 'decp_features.parquet')


def build_app() -> None:
    st.set_page_config(page_title="Marchés publics suspects", layout="wide")

    st.title("Détection d'anomalies dans les marchés publics")

    page = st.sidebar.radio("", ["Contexte", "Données", "Résultats", "Exploration"])

    if page == "Contexte":
        st.markdown("""
        ### Objectif
        Détecter automatiquement les marchés publics français suspects à partir des données
        DECP (data.gouv.fr). On cherche des signaux de favoritisme ou de contournement des règles.

        ### Signaux détectés
        - **Offre unique** : 1 seul candidat
        - **Sans concurrence** : procédure sans appel d'offres
        - **Seuil shaving** : montant juste sous 40 000€
        - **Durée excessive** : contrat > 48 mois

        ### Approche ML
        3 modèles supervisés comparés en cross-validation (5 folds) :
        Régression Logistique, Random Forest, XGBoost.
        """)

    elif page == "Données":
        df = load_data()
        col1, col2, col3 = st.columns(3)
        col1.metric("Marchés analysés", f"{len(df):,}")
        col2.metric("Suspects (labels)", f"{df['label'].sum():,}")
        col3.metric("Taux de suspicion", f"{df['label'].mean()*100:.1f}%")

        st.subheader("Distribution des montants")
        fig = px.histogram(
            df[df['montant'] > 0]['montant'].clip(upper=df['montant'].quantile(0.99)),
            nbins=50
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Procédures utilisées")
        fig2 = px.bar(df['procedure'].value_counts().reset_index(),
                      x='count', y='procedure', orientation='h')
        st.plotly_chart(fig2, use_container_width=True)

    elif page == "Résultats":
        st.subheader("Comparaison des modèles — cross-validation 5 folds")
        if MODEL_METRICS_FILE.exists():
            st.dataframe(pd.read_csv(MODEL_METRICS_FILE), use_container_width=True)
        else:
            st.info("Lance `python scripts/main.py` pour générer les métriques.")

        st.image('../plots/comparaison_modeles.png', use_container_width=True)

    elif page == "Exploration":
        df = load_data()
        st.subheader("Top marchés suspects")
        st.dataframe(
            df[df['label'] == 1][['id', 'objet', 'montant', 'procedure', 'offresRecues', 'dureeMois']]
            .head(50),
            use_container_width=True
        )


if __name__ == "__main__":
    build_app()
