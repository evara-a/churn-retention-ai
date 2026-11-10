"""Generate a synthetic dataset with the SAME schema as the IBM/Kaggle
'Telco Customer Churn' dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv).

If you have the real CSV, just pass it to main.py with --data and skip this.
Missing values are injected on purpose so the cleaning step has real work to do.
"""
import numpy as np
import pandas as pd


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21],
    )
    yn = lambda p: rng.choice(["Yes", "No"], n, p=[p, 1 - p])
    has_net = internet != "No"

    def addon(p):
        vals = rng.choice(["Yes", "No"], n, p=[p, 1 - p]).astype(object)
        vals[~has_net] = "No internet service"
        return vals

    monthly = np.where(internet == "Fiber optic", rng.normal(85, 12, n),
              np.where(internet == "DSL", rng.normal(58, 10, n), rng.normal(22, 5, n)))
    monthly = np.clip(monthly, 18, 120).round(2)

    df = pd.DataFrame({
        "customerID": [f"{i:04d}-{''.join(rng.choice(list('ABCDEFGHIJ'), 5))}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": yn(0.48),
        "Dependents": yn(0.30),
        "tenure": tenure,
        "PhoneService": yn(0.90),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10]),
        "InternetService": internet,
        "OnlineSecurity": addon(0.29),
        "OnlineBackup": addon(0.34),
        "DeviceProtection": addon(0.34),
        "TechSupport": addon(0.29),
        "StreamingTV": addon(0.38),
        "StreamingMovies": addon(0.39),
        "Contract": contract,
        "PaperlessBilling": yn(0.59),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
    })
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"] * rng.normal(1, 0.03, n)).round(2)

    # --- hidden churn logic (what the model has to discover) ---
    z = -1.0
    z += np.where(contract == "Month-to-month", 1.3, np.where(contract == "One year", -0.2, -1.2))
    z += np.where(internet == "Fiber optic", 0.7, np.where(internet == "No", -0.7, 0))
    z += -0.045 * tenure
    z += np.where(payment == "Electronic check", 0.55, 0)
    z += np.where(df["TechSupport"] == "No", 0.4, -0.2)
    z += np.where(df["OnlineSecurity"] == "No", 0.3, -0.2)
    z += 0.3 * df["SeniorCitizen"]
    z += 0.012 * (monthly - 65)
    z += rng.normal(0, 0.6, n)
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")

    # --- inject messy data ---
    # Real Telco data has blank-string TotalCharges for tenure==0 customers
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    for col, frac in [("MonthlyCharges", 0.015), ("InternetService", 0.01),
                      ("Contract", 0.008), ("PaymentMethod", 0.01)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


if __name__ == "__main__":
    out = "data/telco_churn.csv"
    generate().to_csv(out, index=False)
    print(f"Saved {out}")
