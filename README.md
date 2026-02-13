# Causal Effect Estimator with Machine Learning

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)

![Maintained](https://img.shields.io/badge/Maintained%3F-yes-000000.svg?style=for-the-badge)

---

## Overview

Causal Effect Estimator is a comprehensive, production-ready application for estimating causal treatment effects from observational data. It bridges classical statistical estimation (MLE, IPTW) with modern machine learning approaches (Causal Forest, Metalearners, Doubly Robust), all wrapped in a clean, minimalist black-and-white interface.

## Start

### Run with pip

```bash
# Clone the repository
git clone https://github.com/sefabilicier/Casual-Effect-Estimator
cd causal-effect-estimator

# Create virtual environment (Python 3.11 recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Estimation Methods

```text
| Method | Type | Description | CATE? |
|--------|------|------------|-------|
| Linear Regression | Traditional | OLS with treatment indicator | ❌ |
| IPTW | Traditional | Inverse probability of treatment weighting | ❌ |
| Stratification | Traditional | Stratification on propensity scores | ❌ |
| Causal Forest | Tree-based | Random forest for heterogeneous effects | ✅ |
| S-Learner | Metalearner | Single model with treatment as feature | ✅ |
| T-Learner | Metalearner | Separate models for treatment/control | ✅ |
| X-Learner | Metalearner | Cross-fitted CATE models | ✅ |
| Doubly Robust | Robust | Combines outcome + propensity models | ❌ |
| AIPW | Robust | Augmented IPW | ❌ |
| TARNet | Deep Learning | Treatment-agnostic representation network | ✅ |
```
---

## Project Structure

```
causal_effect_estimator/
├── app.py
├── config.py
├── requirements.txt
├── environment.yml
├── Dockerfile
├── README.md
│
├── data/
│   ├── generator.py
│   └── loaders.py
│
├── models/
│   ├── base.py
│   ├── traditional.py
│   ├── ml_based.py
│   ├── robust.py
│   └── diagnostic.py
│
├── evaluation/
│   ├── metrics.py
│   ├── visualization.py
│   └── benchmark.py
│
└── utils/
    └── helpers.py
```

---

## Example Workflow

```python
# 1. Generate synthetic data
generator = CausalDataGenerator(
    n_samples=2000,
    treatment_effect=2.0,
    effect_heterogeneity=True
)
df, true_params = generator.generate()

# 2. Train causal forest
cf = CausalForest(n_estimators=100, max_depth=5)
cf.fit(X, T, Y)
ate = cf.estimate_ate()
ites = cf.get_ite()

# 3. Evaluate
metrics = CausalMetrics.ate_error(ate, true_params['ate'])
print(f"ATE Error: {metrics['ate_absolute_error']:.3f}")

# 4. Check assumptions
diagnostics = CausalDiagnostics.check_overlap(ps, T)
```