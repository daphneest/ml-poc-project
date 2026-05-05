from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any

from config import DATA_DIR


FEATURES = [
    'montant_log', 'dureeMois', 'offresRecues', 'nb_titulaires',
    'procedure_code', 'mois_notification',
    'montant_par_mois', 'z_montant_cpv',
    'taux_sans_conc_acheteur', 'nb_marches_acheteur', 'ratio_montant_acheteur',
    'pca_1', 'pca_2',
]


def _parse(marches):
    rows = []
    for m in marches:
        acheteur = m.get('acheteur', {})
        acheteur_id = acheteur.get('id') if isinstance(acheteur, dict) else None
        titulaires = m.get('titulaires', [])
        nb_titulaires = len(titulaires) if isinstance(titulaires, list) else 0
        rows.append({
            'id': m.get('id'),
            'objet': m.get('objet', ''),
            'acheteur_id': acheteur_id,
            'montant': m.get('montant'),
            'dureeMois': m.get('dureeMois'),
            'offresRecues': m.get('offresRecues'),
            'procedure': m.get('procedure'),
            'codeCPV': str(m.get('codeCPV', ''))[:2],
            'nb_titulaires': nb_titulaires,
            'dateNotification': m.get('dateNotification'),
        })
    return pd.DataFrame(rows)


def _clean(df):
    for col in ['montant', 'dureeMois', 'offresRecues']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['dateNotification'] = pd.to_datetime(df['dateNotification'], errors='coerce')
    return df


def _features(df):
    df['montant_log'] = np.log1p(df['montant'])
    df['offre_unique'] = (df['offresRecues'] == 1).astype(int)
    df['sans_concurrence'] = df['procedure'].str.contains(
        'sans publicité|sans mise en concurrence', case=False, na=False
    ).astype(int)
    df['seuil_shaving'] = ((df['montant'] >= 35_000) & (df['montant'] < 40_000)).astype(int)
    df['duree_longue'] = (df['dureeMois'] > 48).astype(int)
    df['mois_notification'] = df['dateNotification'].dt.month.fillna(0).astype(int)
    df['montant_par_mois'] = df['montant'] / (df['dureeMois'] + 1)
    df['z_montant_cpv'] = df.groupby('codeCPV')['montant'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1)
    )
    procedure_map = {
        'Procédure adaptée': 0,
        "Appel d'offres ouvert": 1,
        'Marché passé sans publicité ni mise en concurrence préalable': 2,
        'Procédure avec négociation': 3,
        "Appel d'offres restreint": 4,
        'Procédure négociée ouverte': 5,
        'Dialogue compétitif': 6,
    }
    df['procedure_code'] = df['procedure'].map(procedure_map).fillna(-1).astype(int)
    df['taux_sans_conc_acheteur'] = df.groupby('acheteur_id')['sans_concurrence'].transform('mean')
    df['nb_marches_acheteur'] = df.groupby('acheteur_id')['id'].transform('count')
    df['montant_moyen_acheteur'] = df.groupby('acheteur_id')['montant'].transform('mean')
    df['ratio_montant_acheteur'] = df['montant'] / (df['montant_moyen_acheteur'] + 1)

    features_pca = ['montant_log', 'dureeMois', 'offresRecues', 'montant_par_mois', 'z_montant_cpv']
    X_scaled = StandardScaler().fit_transform(df[features_pca].fillna(0))
    composantes = PCA(n_components=2).fit_transform(X_scaled)
    df['pca_1'] = composantes[:, 0]
    df['pca_2'] = composantes[:, 1]

    df['label'] = (
        df['offre_unique'] + df['sans_concurrence'] +
        df['seuil_shaving'] + df['duree_longue']
    ).clip(upper=1).astype(int)

    return df


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    with open(DATA_DIR / 'decp-2022.json', 'r') as f:
        data = json.load(f)

    df = _parse(data['marches'])
    df = _clean(df)
    df = _features(df)

    X = df[FEATURES].fillna(0)
    y = df['label']

    return tuple(train_test_split(X, y, test_size=0.2, random_state=42))
