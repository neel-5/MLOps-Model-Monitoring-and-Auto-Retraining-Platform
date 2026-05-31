# API Examples

Base URL:

```text
http://127.0.0.1:8000
```

## Health

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

## List Datasets

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/datasets
```

## Train a Model

```powershell
$body = @{
  dataset_id = 1
  target_column = "churn"
  trigger_reason = "manual API training"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/train `
  -ContentType "application/json" `
  -Body $body
```

## Detect Drift

```powershell
$body = @{
  baseline_dataset_id = 1
  current_dataset_id = 3
  model_version = "v1"
  threshold = 0.2
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/drift `
  -ContentType "application/json" `
  -Body $body
```

## Predict

```powershell
$features = @{
  age = 34
  tenure_months = 8
  monthly_charges = 91.4
  total_charges = 731.2
  support_calls = 3
  contract_type = "Month-to-month"
  internet_service = "Fiber optic"
  payment_method = "Electronic check"
  paperless_billing = "Yes"
}

$body = @{
  model_version = "v1"
  features = $features
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/predict `
  -ContentType "application/json" `
  -Body $body
```

## Run Monitoring Cycle

```powershell
$body = @{
  baseline_dataset_id = 1
  current_dataset_id = 3
  model_version = "v1"
  drift_threshold = 0.2
  degradation_threshold = 0.08
  auto_retrain = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/monitor/run `
  -ContentType "application/json" `
  -Body $body
```

## Create Demo Prediction Traffic

```powershell
$body = @{
  dataset_id = 3
  model_version = "v1"
  limit = 40
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/monitoring/demo-predictions `
  -ContentType "application/json" `
  -Body $body
```

## Monitoring Summary

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/monitoring/summary
```
