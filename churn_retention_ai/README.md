# Customer Churn & Retention AI
**Python · Scikit-learn · Pandas · LangChain**

An end-to-end pipeline that predicts which telecom customers are likely to leave, explains why,
and uses an LLM (via LangChain) to draft personalised retention emails for the highest-risk active customers.

## Pipeline
1. **EDA** (`churn/eda.py`) – shape, missing values, churn rate, plots saved to `outputs/eda_overview.png`
2. **Preprocessing** (`churn/preprocess.py`) – type fixes (blank `TotalCharges`), median / most-frequent
   imputation, scaling, one-hot encoding, engineered `AvgChargePerMonth`. Fitted on train only (no leakage).
3. **Model** (`churn/model.py`) – `RandomForestClassifier` (class-balanced) in a Scikit-learn `Pipeline`,
   stratified 80/20 split, optional `GridSearchCV` (`--tune`). Reports precision/recall/F1, ROC-AUC,
   confusion matrix, ROC curve and feature importance.
4. **Scoring** – out-of-fold churn probability for every customer (5-fold CV), so scores are honest.
5. **LLM retention agent** (`churn/retention.py`) – LangChain chain `prompt | llm | StrOutputParser`.
   Rule-based risk factors + an offer are injected into the prompt so emails are grounded, not hallucinated.
   Falls back to a template if no API key is set.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY or OPENAI_API_KEY
python main.py                # synthetic data is generated automatically
python main.py --data WA_Fn-UseC_-Telco-Customer-Churn.csv --tune --top-n 10   # real Kaggle data
```
Outputs land in `outputs/`: plots, `churn_model.joblib`, `retention_emails.md/.csv`.

## Results (synthetic data, 5,000 customers)
ROC-AUC ≈ 0.84, churn recall ≈ 0.74. Top drivers: tenure, contract type, total/monthly charges, fiber internet.
Numbers on the real Kaggle dataset will differ (typically AUC ≈ 0.84–0.86).

## Ideas to extend
SHAP explanations · XGBoost/LightGBM comparison · threshold tuning by campaign cost ·
Streamlit dashboard · A/B-test framework for the emails.
