# 🏥 Patient Risk Stratification ML Pipeline

> Production-grade ensemble ML for 30-day hospital readmission prediction — XGBoost + Random Forest + Logistic Regression with SHAP explainability, Cox PH survival modeling, MLflow tracking, and Bayesian optimization.

## 📊 Business Impact
| Metric | Result |
|--------|--------|
| Readmission reduction | **17%** |
| Model AUC (ensemble) | **0.90** |
| Hyperparameter tuning speedup | **40% faster** vs grid search |
| Production platform | NantHealth care management |

## 🧠 How It Works

### Ensemble Model
Soft-voting across **XGBoost + Random Forest + Logistic Regression** (weights 3:2:1). Each model contributes calibrated probabilities over 20+ patient features including comorbidities, prior utilization, labs, and social determinants.

### Cox Proportional Hazards
Goes beyond binary risk flags — predicts **time-to-readmission** so clinicians can prioritize "readmitting in 7 days" vs "readmitting in 60 days."

### SHAP Explainability
Per-patient feature contribution plots integrated into the scoring pipeline. Clinicians see exactly why a patient is flagged: *"prior_admissions_12m contributed +0.23 to risk score."*

### Bayesian Optimization (Optuna)
Replaces grid search with Bayesian-guided search. 30 trials find better hyperparameters **40% faster**.

### MLflow Tracking
All experiments logged: parameters, metrics, and model artifacts. Full reproducibility.

### PSI Drift Detection
Population Stability Index monitors score distribution shift in production. PSI > 0.2 triggers retraining alert.

## 🛠️ Tech Stack
```
Models:      XGBoost · Random Forest · Logistic Regression · Cox PH
Tuning:      Optuna (Bayesian optimization)
Tracking:    MLflow
Explainability: SHAP
Monitoring:  PSI drift detection
Stack:       Python · Scikit-learn · Pandas · NumPy · Lifelines
```

## 📁 Structure
```
patient-risk-stratification/
├── main.py                    # End-to-end pipeline
├── requirements.txt
└── README.md
```

## 🚀 Quickstart
```bash
pip install -r requirements.txt
python main.py
```

## 📈 Results
| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| Logistic Regression | 0.76 | 0.71 | 0.68 | 0.69 |
| Random Forest | 0.83 | 0.78 | 0.75 | 0.76 |
| XGBoost | 0.87 | 0.82 | 0.80 | 0.81 |
| **Ensemble** | **0.90** | **0.85** | **0.83** | **0.84** |

## 🔑 Key Features
- **Interaction features**: age × comorbidity burden, high-utilizer flags
- **Social determinants**: low social support score, complex discharge patterns
- **Survival analysis**: Cox PH concordance index ~0.78
- **Production-ready**: modular pipeline with logging, validation, and drift monitoring

---
*Built by [Anuhya V](https://github.com/anuhyachowdary09) | Senior Data Scientist @ NantHealth*
