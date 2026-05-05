from __future__ import annotations

from typing import Any
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        'f1': round(f1_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred), 4),
        'recall': round(recall_score(y_true, y_pred), 4),
        'auc_roc': round(roc_auc_score(y_true, y_pred), 4),
    }
