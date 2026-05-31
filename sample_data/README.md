# Sample Datasets

These CSV files model a telecom customer churn workflow:

- `customer_churn_baseline.csv`: baseline training distribution
- `customer_churn_current.csv`: mildly shifted production-like data
- `customer_churn_drifted.csv`: stronger drift intended to trigger retraining

Regenerate them with:

```powershell
py -3.13 scripts\generate_sample_data.py
```
