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

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Contexte", "Données", "Modèles", "Résultats"])

    if page == "Contexte":
        st.title("Détection d'anomalies dans les marchés publics")
        st.markdown("""
        ### Objectif
        Détecter automatiquement les marchés publics français suspects
        à partir des données DECP open source (data.gouv.fr).

        ### Signaux recherchés
        | Signal | Description |
        |---|---|
        | Offre unique | 1 seul candidat — pas de vraie concurrence |
        | Sans concurrence | Procédure sans appel d'offres |
        | Seuil shaving | Montant juste sous 40 000€ |
        | Durée excessive | Contrat de plus de 48 mois |

        ### Dataset
        - **Source** : data.gouv.fr — DECP consolidé 2022
        - **Volume** : 8 756 marchés publics
        - **Licence** : Open Licence 2.0
        """)

    elif page == "Données":
        df = load_data()
        st.title("Exploration des données")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Marchés", f"{len(df):,}")
        col2.metric("Suspects", f"{df['label'].sum():,}")
        col3.metric("Taux", f"{df['label'].mean()*100:.1f}%")
        col4.metric("Acheteurs uniques", f"{df['acheteur_id'].nunique():,}")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Distribution des montants")
            fig = px.histogram(
                df[df['montant'] > 0]['montant'].clip(upper=df['montant'].quantile(0.99)),
                nbins=50, color_discrete_sequence=['steelblue']
            )
            fig.update_layout(xaxis_title="Montant (€)", yaxis_title="Nb marchés", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Procédures utilisées")
            fig2 = px.bar(
                df['procedure'].value_counts().reset_index(),
                x='count', y='procedure', orientation='h',
                color_discrete_sequence=['steelblue']
            )
            fig2.update_layout(yaxis_title="", xaxis_title="Nb marchés", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Espace PCA — bleu = normal, rouge = suspect")
        sample = df.sample(min(2000, len(df)), random_state=42)
        fig3 = px.scatter(
            sample, x='pca_1', y='pca_2',
            color=sample['label'].map({0: 'Normal', 1: 'Suspect'}),
            color_discrete_map={'Normal': 'steelblue', 'Suspect': 'crimson'},
            opacity=0.5,
        )
        st.plotly_chart(fig3, use_container_width=True)

    elif page == "Modèles":
        st.title("Comparaison des modèles")
        st.markdown("""
        3 modèles supervisés entraînés et comparés en **cross-validation 5 folds**.
        Le label utilisé : marché suspect si au moins 1 signal règle métier activé.
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Régression Logistique**\nModèle de base linéaire. Rapide et interprétable.")
        with col2:
            st.info("**Random Forest**\nEnsemble d'arbres de décision. Robuste aux outliers.")
        with col3:
            st.info("**XGBoost**\nGradient boosting. Meilleur modèle sélectionné.")

        st.subheader("Feature Engineering — 4 catégories")
        st.markdown("""
        | Catégorie | Features |
        |---|---|
        | Métier | montant_log, procedure_code, mois_notification |
        | Mathématiques | montant_par_mois, z_montant_cpv |
        | Agrégation | taux_sans_conc_acheteur, nb_marches_acheteur, ratio_montant_acheteur |
        | PCA | pca_1, pca_2 |
        """)

    elif page == "Résultats":
        st.title("Résultats")

        if MODEL_METRICS_FILE.exists():
            st.subheader("Métriques sur le jeu de test")
            st.dataframe(pd.read_csv(MODEL_METRICS_FILE), use_container_width=True)

        st.subheader("Comparaison des modèles")
        st.image('../plots/comparaison_modeles.png', use_container_width=True)

        df = load_data()
        st.subheader("Top 20 marchés les plus suspects")
        st.dataframe(
            df[df['label'] == 1][['id', 'objet', 'montant', 'procedure', 'offresRecues', 'dureeMois']]
            .sort_values('montant', ascending=False)
            .head(20)
            .reset_index(drop=True),
            use_container_width=True
        )


if __name__ == "__main__":
    build_app()
