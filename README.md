# 📊 Healthcare Predictive Analytics – Stroke Risk Prediction

A data science and machine learning project focused on building a healthcare predictive analytics system for early stroke risk assessment. By analyzing patient demographic, clinical, and lifestyle data, the system identifies patterns associated with potential health risks to support clinical decision-making.

---

## 📖 Table of Contents

- [💡 Overview](#-overview)
- [🎯 Objectives](#-objectives)
- [👥 Team & Roles](#-team--roles)
- [📌 Scope](#-scope)
- [📁 Project Structure](#-project-structure)
- [📋 Requirements](#-requirements)
- [🚀 Run the App](#-run-the-app)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 💡 Overview

Stroke is one of the leading causes of death and long-term disability worldwide, and early identification of at-risk patients can significantly improve outcomes. This project aims to develop a healthcare predictive analytics system that analyzes patient demographic, clinical, and lifestyle data to identify patterns associated with stroke risk — enhancing early risk assessment and supporting clinical decision-making.

This repository includes the project proposal and documentation, along with the data pipeline, trained model, a Streamlit demo app, and a FastAPI service for real-time predictions.

---

## 🎯 Objectives

- **Risk Factor Analysis:** Analyze healthcare data to identify key risk factors associated with stroke.
- **Exploratory Data Analysis (EDA):** Understand medical and lifestyle patterns in the dataset.
- **Model Development:** Build and evaluate machine learning models for stroke risk prediction.
- **Interpretability:** Deliver an accurate and interpretable healthcare prediction system.
- **Clinical Relevance:** Provide insights that can support early intervention and preventive care.

---

## 👥 Team & Roles

Each member owned an end-to-end stage of the pipeline — every contribution is demonstrable at the final presentation.

| Member | Role | Contribution |
|---|---|---|
| **Muhammed Salah** | Team Leader | Project management, requirements & final documentation |
| **George Emil** | ML Engineer | Model development & optimization (Milestone 3) |
| **Omar Ahmed** | Data Engineer | Data collection, cleaning & preprocessing (Milestone 1) |
| **Toqa Hamed** | Data Analyst | Exploratory data analysis & visualization (Milestone 2) |
| **Mostafa Mohamed** | Backend / MLOps Engineer | API, deployment & monitoring (Milestone 4) |
| **Nermine Samir** | QA & Presentation Lead | Testing, validation & final presentation (Milestone 5) |

---

## 📌 Scope

- **Data Collection:** Patient demographic, clinical, and lifestyle data relevant to stroke risk.
- **EDA:** Identify trends, correlations, and influential features in the dataset.
- **Model Development:** Implement and compare machine learning models (e.g., Logistic Regression, Random Forest, XGBoost) for stroke risk classification.
- **Evaluation:** Assess model performance using metrics such as accuracy, precision, recall, and ROC-AUC.
- **Deployment (planned):** Package the final model into an accessible tool for demonstration purposes.

---

## 📁 Project Structure

```
Healthcare-Predictive-Analytics-Project-Proposal/
├── api/                                                 # FastAPI app for real-time predictions
├── src/                                                 # Reusable training and prediction code
├── dataset/                                             # Raw and processed datasets
│   └── archive/                                         # Unused alternate source data, kept for traceability
├── docs/                                                # Project and assignment PDFs
├── models/                                              # Canonical trained model artifact
├── Notebooks/                                           # EDA and modeling notebooks
├── reports/                                              # Evaluation reports, metrics, and plots
├── streamlit_app.py                                     # Streamlit demo app
├── requirements.txt                                     # Python dependencies
├── Healthcare_Predictive_Analytics_Proposal2 (1).pdf    # Project proposal
├── Healthcare_Stroke_Risk_Prediction_Project Doc.pdf    # Detailed project documentation
├── README.md                                            # Project documentation
```

**Canonical model:** `models/stroke_model_pipeline.joblib` — contains the scikit-learn preprocessing pipeline, selected model, decision threshold, feature list, and metadata.

---

## 📋 Requirements

- Python 3.8 or higher
- FastAPI, Uvicorn, Pydantic
- NumPy, Pandas, Scikit-learn, Joblib
- Matplotlib, Seaborn
- MLflow
- Streamlit

Install everything with:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

The trained model can be served two ways: as an interactive **Streamlit** app, or as a **FastAPI** REST API.

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/nermeenesamir/Healthcare-Predictive-Analytics-Project-Proposal.git
cd Healthcare-Predictive-Analytics-Project-Proposal
pip install -r requirements.txt
```

### 2. Run the Streamlit demo app

```bash
python -m streamlit run streamlit_app.py
```

This opens an interactive form in your browser where you can enter patient details (age, hypertension, heart disease, glucose level, BMI, smoking status, etc.) and get a stroke risk prediction with probability.

### 3. (Optional) Run the FastAPI service

```bash
uvicorn api.main:app --reload
```

Then open the interactive API docs at:

```
http://127.0.0.1:8000/docs
```

> ⚠️ The `bmi_missing` feature is generated automatically by `src.predict` — you don't need to send it manually when calling the API.

### 4. (Optional) Reproduce model training

```bash
python -m src.train
```

To log runs with MLflow:

```bash
python -m src.train --mlflow
```

---

## 🗺️ Roadmap

- [x] Project proposal and documentation
- [x] Data collection and preprocessing
- [x] Exploratory data analysis
- [x] Model development and evaluation
- [x] Streamlit demo app and FastAPI service
- [ ] Final report and results
- [ ] Live cloud deployment

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
