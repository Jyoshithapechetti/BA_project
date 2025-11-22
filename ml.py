# ml.py
# Lightweight ML: TF-IDF → (LogReg classifier, Ridge regressor)
# Trains on your loaded DataFrame (columns: "review", "rating") and exposes prediction helpers.

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)

_CLEAN_RE = re.compile(r"<.*?>")

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _CLEAN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _make_label_from_rating(rating: float, threshold: float = 7.0) -> int:
    """>= threshold → Positive (1), else Negative (0)."""
    try:
        return int(float(rating) >= float(threshold))
    except Exception:
        return 0

@dataclass
class TrainedModels:
    clf_pipeline: Pipeline
    reg_pipeline: Pipeline
    threshold: float
    metrics: Dict[str, Any]
    # store splits for potential debugging
    n_train: int
    n_test: int


# Public API

def train_models(
    df: pd.DataFrame,
    *,
    threshold: float = 7.0,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 50000,
) -> TrainedModels:
    """
    Train sentiment classifier (Positive/Negative) and rating regressor.
    Returns pipelines and hold-out metrics.
    """
    if "review" not in df.columns or "rating" not in df.columns:
        raise ValueError("DataFrame must contain 'review' and 'rating' columns.")

    dd = df[["review", "rating"]].dropna().copy()
    dd["review"] = dd["review"].apply(_clean_text)
    dd = dd[dd["review"].str.len() > 0]

    # labels/targets
    y_cls = dd["rating"].apply(lambda r: _make_label_from_rating(r, threshold)).astype(int)
    y_reg = pd.to_numeric(dd["rating"], errors="coerce")

    # Split once; reuse indices for both tasks so metrics compare on same hold-out
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        dd["review"].values, y_cls.values, y_reg.values,
        test_size=test_size, random_state=random_state, stratify=y_cls
    )

    # Pipelines
    tfidf_cfg = dict(max_features=max_features, ngram_range=(1, 2), min_df=3)

    clf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_cfg)),
        ("clf", LogisticRegression(solver="liblinear", max_iter=1000, class_weight="balanced")),
    ])

    reg_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_cfg)),
        ("reg", Ridge(alpha=1.0, random_state=random_state)),
    ])

    # Fit
    clf_pipeline.fit(X_train, y_cls_train)
    reg_pipeline.fit(X_train, y_reg_train)

    # Metrics (hold-out)
    cls_pred = clf_pipeline.predict(X_test)
    try:
        cls_proba = clf_pipeline.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_cls_test, cls_proba)
    except Exception:
        cls_proba = None
        roc = np.nan

    acc = accuracy_score(y_cls_test, cls_pred)
    prec = precision_score(y_cls_test, cls_pred, zero_division=0)
    rec = recall_score(y_cls_test, cls_pred, zero_division=0)
    f1 = f1_score(y_cls_test, cls_pred, zero_division=0)
    cm = confusion_matrix(y_cls_test, cls_pred).tolist()  # for JSON friendliness

    reg_pred = reg_pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_reg_test, reg_pred)))
    mae = float(mean_absolute_error(y_reg_test, reg_pred))
    r2  = float(r2_score(y_reg_test, reg_pred))

    metrics = {
        "classification": {
            "threshold": threshold,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc) if roc == roc else None,  # None if NaN
            "confusion_matrix": cm,  # [[tn, fp],[fn,tp]]
            "n_test": int(len(X_test)),
        },
        "regression": {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_test": int(len(X_test)),
        },
    }

    return TrainedModels(
        clf_pipeline=clf_pipeline,
        reg_pipeline=reg_pipeline,
        threshold=threshold,
        metrics=metrics,
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
    )

def predict_text(text: str, models: TrainedModels) -> Dict[str, Any]:
    """
    Predict sentiment (Positive/Negative + probability if available) and rating.
    """
    txt = _clean_text(text)
    if not txt:
        return {"label": None, "prob_positive": None, "pred_rating": None}

    # Classification
    label_bin = int(models.clf_pipeline.predict([txt])[0])
    label = "Positive" if label_bin == 1 else "Negative"
    prob_pos = None
    try:
        prob_pos = float(models.clf_pipeline.predict_proba([txt])[0, 1])
    except Exception:
        prob_pos = None

    # Regression
    raw_pred = float(models.reg_pipeline.predict([txt])[0])
    pred_rating = max(1.0, min(10.0, raw_pred))


    return {
        "label": label,
        "prob_positive": prob_pos,
        "pred_rating": pred_rating,
    }

def get_metrics(models: TrainedModels) -> Dict[str, Any]:
    """Return metrics dict computed at train time."""
    return models.metrics
