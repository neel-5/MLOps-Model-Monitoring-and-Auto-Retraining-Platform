from __future__ import annotations

import csv
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "sample_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def choice_weighted(rng: random.Random, choices: list[tuple[str, float]]) -> str:
    total = sum(weight for _, weight in choices)
    cursor = rng.random() * total
    upto = 0.0
    for label, weight in choices:
        upto += weight
        if upto >= cursor:
            return label
    return choices[-1][0]


def bounded_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def build_rows(seed: int, size: int, mode: str) -> list[dict]:
    rng = random.Random(seed)
    rows = []

    if mode == "baseline":
        contract_weights = [("Month-to-month", 0.45), ("One year", 0.32), ("Two year", 0.23)]
        payment_weights = [("Electronic check", 0.34), ("Credit card", 0.31), ("Bank transfer", 0.24), ("Mailed check", 0.11)]
        age_shift, tenure_shift, charge_shift, support_shift = 0, 0, 0, 0
    elif mode == "current":
        contract_weights = [("Month-to-month", 0.51), ("One year", 0.30), ("Two year", 0.19)]
        payment_weights = [("Electronic check", 0.40), ("Credit card", 0.27), ("Bank transfer", 0.23), ("Mailed check", 0.10)]
        age_shift, tenure_shift, charge_shift, support_shift = 1, -3, 4, 0.3
    else:
        contract_weights = [("Month-to-month", 0.70), ("One year", 0.20), ("Two year", 0.10)]
        payment_weights = [("Electronic check", 0.57), ("Credit card", 0.18), ("Bank transfer", 0.17), ("Mailed check", 0.08)]
        age_shift, tenure_shift, charge_shift, support_shift = -4, -11, 15, 1.1

    internet_weights = [("Fiber optic", 0.43), ("DSL", 0.38), ("No", 0.19)]

    for index in range(size):
        age = bounded_int(rng.gauss(43 + age_shift, 13), 18, 78)
        tenure = bounded_int(rng.expovariate(1 / 28) + tenure_shift, 1, 72)
        monthly = round(max(20, rng.gauss(69 + charge_shift, 18)), 2)
        support_calls = max(0, int(rng.poisson(lam=1.2 + support_shift)) if hasattr(rng, "poisson") else int(rng.expovariate(1 / (1.3 + support_shift))))
        contract = choice_weighted(rng, contract_weights)
        internet = choice_weighted(rng, internet_weights)
        payment = choice_weighted(rng, payment_weights)
        paperless = choice_weighted(rng, [("Yes", 0.64), ("No", 0.36)])

        logit = -1.55
        logit += 0.028 * (monthly - 70)
        logit -= 0.045 * tenure
        logit += 0.23 * support_calls
        logit += 0.62 if contract == "Month-to-month" else -0.22 if contract == "One year" else -0.55
        logit += 0.38 if internet == "Fiber optic" else -0.18 if internet == "No" else 0
        logit += 0.36 if payment == "Electronic check" else -0.12
        logit += 0.15 if paperless == "Yes" else -0.08
        if mode == "drifted":
            logit += 0.36

        churn_probability = sigmoid(logit)
        churn = 1 if rng.random() < churn_probability else 0
        total = round(max(monthly, monthly * tenure + rng.gauss(0, 65)), 2)

        rows.append(
            {
                "customer_id": f"CUST-{mode[:3].upper()}-{index + 1:04d}",
                "age": age,
                "tenure_months": tenure,
                "monthly_charges": monthly,
                "total_charges": total,
                "support_calls": support_calls,
                "contract_type": contract,
                "internet_service": internet,
                "payment_method": payment,
                "paperless_billing": paperless,
                "churn": churn,
            }
        )
    return rows


def write_dataset(filename: str, rows: list[dict]) -> None:
    path = OUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def main() -> None:
    write_dataset("customer_churn_baseline.csv", build_rows(42, 720, "baseline"))
    write_dataset("customer_churn_current.csv", build_rows(84, 420, "current"))
    write_dataset("customer_churn_drifted.csv", build_rows(126, 420, "drifted"))


if __name__ == "__main__":
    main()
