# 📊 Healthcare Predictive Analytics Project — Stroke Risk Prediction

A machine learning project aimed at developing a healthcare predictive analytics system that processes patient demographic, clinical, and lifestyle data to predict stroke risk, identify trends, and support clinical decision-making.

> Educational project output. Not medical advice.

---

## 📖 Table of Contents

- [💡 Overview](#-overview)
- [🎯 Objectives](#-objectives)
- [📌 Scope](#-scope)
- [🗂️ Dataset](#️-dataset)
- [🧪 Modeling Results](#-modeling-results)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [📋 Requirements](#-requirements)
- [👥 Team](#-team)
- [⚠️ Limitations](#️-limitations)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 💡 Overview

Stroke is one of the leading causes of death and long-term disability worldwide. Early identification of at-risk patients enables preventive intervention. This project builds a predictive system that estimates a patient's stroke risk from routine demographic, clinical, and lifestyle data — and exposes it through a REST API and web app, not left as a notebook exercise.

The system estimates a patient's risk of stroke using demographic factors (age, gender, marital status), clinical indicators (hypertension, heart disease, average glucose level, BMI), and lifestyle factors (work type, residence type, smoking status), to support early screening and triage rather than replace clinical diagnosis.

---

## 🎯 Objectives

- **Stroke Risk Prediction:** Build and compare machine learning classifiers (Logistic Regression, Random Forest, Naive Bayes, SVM, Decision Tree, KNN) that estimate the probability of stroke from routinely collected patient data.
- **Imbalance-Aware Evaluation:** Because only ~4.9% of records are positive for stroke, prioritize Recall, F1-score, ROC-AUC, and PR-AUC over raw accuracy, which is misleading on imbalanced clinical data.
- **Leak-Free Preprocessing:** Encapsulate all preprocessing (imputation, scaling, one-hot encoding) inside a single scikit-learn `Pipeline` fitted only on the training split, so no information from the test set leaks into training.
- **Threshold Optimization:** Move beyond the default 0.5 decision threshold and tune it to balance false negatives (missed strokes) against false positives (unnecessary alerts).
- **Deployable Artifact:** Persist the final pipeline (preprocessing + model) as a single serialized object loaded by a lightweight web application and REST API.

---

## 📌 Scope

- **Data Collection:** The public Kaggle "Stroke Prediction Dataset" — 5,110 patient records, 12 columns.
- **EDA (Exploratory Data Analysis):** Distribution analysis, class-balance checks, bivariate analysis against the target, correlation analysis, and missing-value diagnostics.
- **Model Development:** Stratified 60/20/20 train/validation/test split, and comparison of six classifiers using class-imbalance-aware metrics.
- **Deployment:** Local Streamlit application and FastAPI endpoints for real-time and batch predictions.
- **Clinical scope:** Educational decision-support prototype; outputs are not medical advice and require stakeholder validation before production use.

---

## 🗂️ Dataset

| Item | Value |
|---|---|
| Source | Kaggle ["Stroke Prediction Dataset"](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (`healthcare-dataset-stroke-data.csv`) |
| Rows | 5,110 |
| Raw columns | 12 |
| Stroke cases | 249 (4.87%) |
| No-stroke cases | 4,861 (95.13%) |
| Missing BMI rows | 201 (3.93%) |

Only `bmi` has missing values. Missingness is informative — the stroke rate is **19.90%** among patients with a missing BMI, versus **4.26%** among patients with a recorded BMI — so the pipeline applies median imputation plus a `bmi_missing` indicator rather than dropping or silently imputing.

Age shows the strongest linear relationship with stroke (r = 0.245), followed by heart disease (r = 0.135), average glucose level (r = 0.132), and hypertension (r = 0.128). BMI alone is only weakly correlated (r = 0.042).

---

## 🧪 Modeling Results

Six classifiers were trained inside identical leak-free preprocessing pipelines and compared on the validation set using ROC-AUC and PR-AUC. **Logistic Regression** was selected as the final model — it led on validation ROC-AUC (0.842) and PR-AUC (0.210), and remains fully interpretable, which matters for a healthcare screening tool.

### Final test-set performance (tuned threshold = 0.8028)

| Metric | Value |
|---|---|
| Accuracy | 90.90% |
| Precision (stroke class) | 24.71% |
| Recall (stroke class) | 42.00% |
| F1 (stroke class) | 31.11% |
| ROC-AUC | 0.840 |
| PR-AUC (Average Precision) | 0.238 |

Out of 1,022 held-out test patients, the model correctly flagged 21 of the 50 true stroke cases while producing 64 false positives among 972 non-stroke patients — the expected precision/recall trade-off on a rare-event healthcare problem. Full EDA figures, the six-model comparison table, evaluation curves, and functional test results are in [`reports/Final_Project_Report.pdf`](reports/Final_Project_Report.pdf).

---

## ⚙️ Installation

```bash
# Clone the repository
$ git clone https://github.com/nermeenesamir/Healthcare-Predictive-Analytics-Project-Proposal.git

# Navigate to the project directory
$ cd Healthcare-Predictive-Analytics-Project-Proposal

# Install the required packages
$ pip install -r requirements.txt
```

---

## 🚀 Usage

- Run the Jupyter notebooks in `notebooks/` for data exploration and model comparison.
- Retrain the model from raw data:

```bash
python -m src.train
```

- Run the REST API:

```bash
uvicorn api.main:app --reload
```

- Run the web app (in a separate terminal):

```bash
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
Healthcare-Predictive-Analytics-Project-Proposal/
├── api/
│   └── main.py                  # FastAPI app — /health, /predict, /predict-batch
├── src/
│   ├── train.py                 # End-to-end training script (data → pipeline → 6-model comparison → threshold tuning → artifact)
│   └── predict.py               # Shared StrokePredictor class used by the API and the Streamlit app
├── streamlit_app.py             # Web application entry point
├── notebooks/
│   ├── EDA_Notebook.ipynb       # Exploratory data analysis (Milestones 1–2)
│   └── Modeling_Notebook.ipynb  # Model comparison notebook (Milestone 3)
├── dataset/                     # Raw and processed CSV data
├── models/
│   └── stroke_model_pipeline.joblib   # Serialized preprocessing + model pipeline
├── reports/
│   ├── Final_Project_Report.pdf
│   └── Stroke_Risk_Prediction_Presentation.pdf
├── requirements.txt              # Required Python packages
└── README.md                     # Project documentation
```

---

## 📋 Requirements

- Python 3.10 or higher
- pandas, numpy, scikit-learn, joblib
- matplotlib, seaborn
- fastapi, uvicorn, pydantic
- streamlit
- mlflow (experiment tracking, planned)

Full pinned versions in [`requirements.txt`](requirements.txt).

---

## 👥 Team — Group Code CAI_AIS4_S2

| Member | Role |
|---|---|
| Muuhmd Salah Abd El-Bari | Team Leader — project management, requirements & final documentation |
| Omar Ahmed Mostafa | Data Engineer — data collection, cleaning & preprocessing (Milestone 1) |
| Toqa Hamed Mohamed | Data Analyst — exploratory data analysis & visualization (Milestone 2) |
| Goerge Emil Sadeq | ML Engineer — model development & optimization (Milestone 3) |
| Mostafa Mohamed Foaad | Backend / MLOps Engineer — API, deployment & monitoring (Milestone 4) |
| Nermine Samir Abd El-Aty | QA & Presentation Lead — testing, validation & final presentation (Milestone 5) |

**Instructor:** Ahmed Mostafa · **Program:** Digital Egypt Pioneers Initiative (DEPI), Round 4, AI & Data Science Track

---

## ⚠️ Limitations

- Modest precision (24.7%) at the tuned threshold — roughly 3 of 4 flagged patients will not go on to have a stroke; the operating threshold should ultimately be set with clinical stakeholders.
- Age dominates the model's score, so very elderly patients score elevated risk almost regardless of other factors.
- Only 249 positive (stroke) cases in the full dataset limits how tightly test-set metrics generalize.
- Resampling techniques (e.g. SMOTE) and hyperparameter tuning were scoped but not yet applied.
- Deployment is currently local-only; cloud hosting, authentication, and drift monitoring are next steps.

This system is a decision-support prototype for educational purposes and must not be relied upon as the sole basis for medical diagnosis or treatment decisions.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request.

---

## 📄 License

Add a license of your choice (e.g. MIT) if this repository is intended to be public.
