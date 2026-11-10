"""LangChain layer: turn high-risk customers into personalised retention emails.

Works in two modes:
  * LLM mode  - uses LangChain + Anthropic/OpenAI when an API key is configured.
  * Offline   - deterministic template fallback, so the pipeline always runs.
"""
import os
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = (
    "You are a customer-retention specialist at a telecom company called ConnectPlus. "
    "Write warm, concise, honest emails. Never invent discounts beyond the offer provided, "
    "never mention 'churn', 'AI', or 'risk score' to the customer, and do not sound pushy."
)

HUMAN_PROMPT = """Write a retention email (max 120 words) for this customer.

Customer ID: {customer_id}
Tenure: {tenure} months
Contract: {contract}
Internet service: {internet}
Monthly bill: {monthly}
Key friction points detected: {risk_factors}
Offer to include: {offer}

Return exactly:
Subject: <subject line>

<email body, signed 'The ConnectPlus Care Team'>"""


# ---------- explainable risk factors + offers (rule-based, grounded in the data) ----------
def risk_factors(row: pd.Series) -> list[str]:
    f = []
    if row.get("Contract") == "Month-to-month":
        f.append("on a flexible month-to-month plan with no long-term commitment")
    if row.get("tenure", 99) < 12:
        f.append("relatively new customer (under 12 months)")
    if row.get("PaymentMethod") == "Electronic check":
        f.append("pays by electronic check (higher payment friction)")
    if row.get("InternetService") != "No" and row.get("TechSupport") == "No":
        f.append("no tech support add-on")
    if row.get("InternetService") != "No" and row.get("OnlineSecurity") == "No":
        f.append("no online security add-on")
    if row.get("MonthlyCharges", 0) > 80:
        f.append("high monthly bill")
    return f or ["no single dominant factor"]


def pick_offer(row: pd.Series) -> str:
    if row.get("Contract") == "Month-to-month":
        return "15% off for 12 months when switching to a 1-year plan"
    if row.get("InternetService") != "No" and row.get("TechSupport") == "No":
        return "3 months of free Premium Tech Support"
    if row.get("MonthlyCharges", 0) > 80:
        return "a free plan review and a 10% loyalty bill credit"
    return "a complimentary service check-up and a $10 loyalty credit"


# ---------- LangChain chain ----------
def build_chain():
    """Return a LangChain runnable (prompt | llm | parser) or None if unavailable."""
    from dotenv import load_dotenv
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0.6)
        elif provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"), temperature=0.6, max_tokens=400)
        else:
            print(f"[retention] No API key for provider '{provider}' -> using offline template mode.")
            return None
    except ImportError as e:
        print(f"[retention] LangChain packages missing ({e}) -> using offline template mode.")
        return None

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return prompt | llm | StrOutputParser()


def _template_email(v: dict) -> str:
    return (
        "Subject: A thank-you from ConnectPlus (and something for you)\n\n"
        f"Hi there,\n\nThank you for being with ConnectPlus for {v['tenure']} months. "
        "We'd love to make your experience even better, so we've put together a little something for you: "
        f"{v['offer']}.\n\nIf anything about your service hasn't been ideal, just reply to this email "
        "and a real person will help you personally.\n\nThe ConnectPlus Care Team"
    )


def generate_emails(scored: pd.DataFrame, top_n: int = 5, threshold: float = 0.5,
                    out_dir: str = "outputs") -> pd.DataFrame:
    """Pick the top-N still-active customers above the risk threshold and write emails."""
    active = scored[(scored["Churn"] == 0) & (scored["churn_probability"] >= threshold)]
    targets = active.head(top_n)
    print(f"\n=== Retention campaign ===\n{len(active)} active customers above {threshold:.0%} risk; "
          f"drafting emails for the top {len(targets)}.")

    chain = build_chain()
    rows = []
    for _, r in targets.iterrows():
        factors = risk_factors(r)
        payload = {
            "customer_id": r["customerID"], "tenure": int(r["tenure"]), "contract": r["Contract"] if pd.notna(r["Contract"]) else "unknown",
            "internet": r["InternetService"] if pd.notna(r["InternetService"]) else "unknown", "monthly": "n/a" if pd.isna(r["MonthlyCharges"]) else f"${r['MonthlyCharges']:.2f}",
            "risk_factors": "; ".join(factors), "offer": pick_offer(r),
        }
        try:
            email = chain.invoke(payload) if chain else _template_email(payload)
        except Exception as e:  # network/key/rate-limit problems shouldn't kill the run
            print(f"[retention] LLM call failed for {payload['customer_id']}: {e}. Using template.")
            email = _template_email(payload)
        rows.append({**{k: payload[k] for k in ("customer_id", "contract", "offer")},
                     "churn_probability": round(float(r["churn_probability"]), 3),
                     "risk_factors": payload["risk_factors"], "email": email})

    result = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    result.to_csv(out / "retention_emails.csv", index=False)
    with open(out / "retention_emails.md", "w") as fh:
        for _, r in result.iterrows():
            fh.write(f"## {r.customer_id}  (churn risk {r.churn_probability:.0%})\n"
                     f"**Why flagged:** {r.risk_factors}\n\n{r.email}\n\n---\n\n")
    print(f"Saved {out / 'retention_emails.md'} and .csv")
    return result
