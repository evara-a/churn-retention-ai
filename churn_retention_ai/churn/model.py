"""Train + evaluate a Random Forest churn classifier."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
                             roc_auc_score)
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from .preprocess import build_preprocessor, split_xy


def train_model(df: pd.DataFrame, out_dir: str = "outputs", tune: bool = False, seed: int = 42):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)

    pipe = Pipeline([
        ("prep", build_preprocessor(X)),
        ("rf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=seed)),
    ])

    if tune:
        grid = {"rf__n_estimators": [200, 400], "rf__max_depth": [None, 12, 20],
                "rf__min_samples_leaf": [3, 5, 10]}
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1)
        search.fit(X_train, y_train)
        print(f"Best params: {search.best_params_}  (CV AUC {search.best_score_:.3f})")
        pipe = search.best_estimator_
    else:
        pipe.fit(X_train, y_train)

    # ---- evaluation ----
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("\n=== Model evaluation (hold-out test set) ===")
    print(classification_report(y_test, pred, target_names=["Stayed", "Churned"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stayed", "Churned"], ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion matrix")
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1])
    axes[1].set_title("ROC curve")
    plt.tight_layout()
    fig.savefig(out / "evaluation.png", dpi=130)
    plt.close(fig)

    # ---- feature importance ----
    importances = feature_importances(pipe)
    top = importances.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color="#3d5a80")
    ax.set_title("Top churn drivers (Random Forest importance)")
    plt.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=130)
    plt.close(fig)
    print("\nTop drivers:\n" + importances.head(8).round(3).to_string())

    joblib.dump(pipe, out / "churn_model.joblib")
    return pipe, X_test, y_test, importances


def feature_importances(pipe: Pipeline) -> pd.Series:
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pd.Series(pipe.named_steps["rf"].feature_importances_, index=names)
    return imp.sort_values(ascending=False)


def score_customers(pipe: Pipeline, df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Attach an out-of-fold churn probability to every customer (highest risk first).

    Out-of-fold means each customer is scored by a model that never saw them in
    training, so the probabilities are honest rather than memorised.
    """
    X, y = split_xy(df)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    proba = cross_val_predict(clone(pipe), X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    out = df.copy()
    out["churn_probability"] = proba
    return out.sort_values("churn_probability", ascending=False)
