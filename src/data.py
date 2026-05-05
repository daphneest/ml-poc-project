from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Any

from config import DATA_DIR


FEATURES = [
    'montant_log', 'dureeMois', 'offresRecues', 'nb_titulaires',
    'procedure_code', 'mois_notification',
    'montant_par_mois', 'z_montant_cpv',
    'taux_sans_conc_acheteur', 'nb_marches_acheteur', 'ratio_montant_acheteur',
    'pca_1', 'pca_2',
]


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    df = pd.read_parquet(DATA_DIR / 'decp_features.parquet')

    X = df[FEATURES].fillna(0)
    y = df['label']

    return tuple(train_test_split(X, y, test_size=0.2, random_state=42))
