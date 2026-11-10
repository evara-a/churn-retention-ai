"""End-to-end pipeline:  data -> EDA -> clean -> Random Forest -> LLM retention emails.

Usage:
    python main.py                      # generates synthetic data if none given
    python main.py --data path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
    python main.py --tune --top-n 10
"""
import argparse
from pathlib import Path

import pandas as pd

from churn.eda import run_eda
from churn.model import score_customers, train_model
from churn.preprocess import clean
from churn.retention import generate_emails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/telco_churn.csv")
    ap.add_argument("--tune", action="store_true", help="GridSearchCV hyper-parameter tuning")
    ap.add_argument("--top-n", type=int, default=5, help="how many high-risk customers get an email")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if not Path(args.data).exists():
        from data.generate_data import generate
        print(f"{args.data} not found -> generating synthetic Telco-style data.")
        Path(args.data).parent.mkdir(exist_ok=True)
        generate().to_csv(args.data, index=False)

    raw = pd.read_csv(args.data)
    run_eda(raw)
    df = clean(raw)
    pipe, *_ = train_model(df, tune=args.tune)
    scored = score_customers(pipe, df)
    generate_emails(scored, top_n=args.top_n, threshold=args.threshold)


if __name__ == "__main__":
    main()
