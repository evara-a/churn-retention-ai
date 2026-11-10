"""Cleaning, missing-value handling and feature engineering."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COL = "customerID"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "AvgChargePerMonth"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that is safe to do before the train/test split."""
    df = df.copy()
    df = df.drop_duplicates(subset=ID_COL) if ID_COL in df else df.drop_duplicates()

    # Telco quirk: TotalCharges is a string with blanks for brand-new customers
    for col in ["TotalCharges", "MonthlyCharges"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip().replace("", np.nan), errors="coerce")

    # Brand-new customers (tenure 0) have not been billed yet -> total is 0, not "unknown"
    df.loc[(df["tenure"] == 0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0

    # Feature engineering
    df["AvgChargePerMonth"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)

    # Binary target
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Imputation + scaling + encoding. Fitted on train only (no leakage)."""
    numeric = [c for c in NUMERIC if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)])


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    y = df[TARGET]
    return X, y
