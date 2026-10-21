# 🏥 Healthcare Risk Classification System

A machine-learning web app that predicts a patient's health risk level (**LOW / MEDIUM / HIGH / CRITICAL**) from clinical and lifestyle data, and explains each prediction with SHAP.

> ⚠️ **Educational project, not a medical device.** It is trained on a synthetic dataset and must not be used for real diagnosis or treatment decisions. Recommendations shown are simple rule-based suggestions.

## Features

- Streamlit app with login, dashboard, patient lookup and manual assessment
- Automatic data cleaning (column renaming, label standardisation, missing values, duplicates, outlier clipping)
- Feature engineering: MAP, BMI-Age index, metabolic risk score, activity score, sleep quality
- Logistic Regression classifier, selected over a Random Forest by cross-validation
- Explainable AI: SHAP shows which factors push risk **higher** (orange) or **lower** (green)
- PDF report download per patient

## Model

| Step | Detail |
|---|---|
| Split | 80/20 stratified, `random_state=42` |
| Preprocessing | StandardScaler (numeric) + OneHotEncoder (categorical), **fit on training data only** |
| Models compared | Logistic Regression vs Random Forest (200 trees, balanced class weights) |
| Validation | 5-fold stratified cross-validation |
| Explainability | SHAP `LinearExplainer`, impact measured on risk severity (LOW=0 … CRITICAL=3) |

### Results

Run `python train_model.py` and copy the printed values here (also saved in `models/metrics.json`):

| Model | Test accuracy | Macro F1 | CV accuracy (mean ± std) |
|---|---|---|---|
| **Logistic Regression (final)** | **0.95** | **0.95** | **0.933 ± 0.025** |
| Random Forest | 0.66 | 0.65 | 0.660 ± 0.014 |

Logistic Regression clearly beat Random Forest (95% vs 66%), which suggests the synthetic labels follow a mostly linear rule of the input features. The model with the best cross-validation accuracy is saved automatically.

### Limitations

- The dataset is **synthetic and perfectly balanced** (250 records per class); the engineered features show little separation between classes, so results will not transfer to real patients.
- `Activity_Score` and `Metabolic_Risk_Score` use assumed formulas, not clinical standards.
- Labels may not follow real clinical definitions; no external validation was performed.
- For a clinical use, the cost of predicting LOW for a CRITICAL patient is high, so per-class recall matters more than overall accuracy.

## Project Structure

```
Health-Risk-Classification/
├── app.py                  # Streamlit application
├── train_model.py          # Trains and saves the model
├── requirements.txt
├── README.md
├── data/
│   └── Healthcare_Risk_Classification_Dataset_Balanced.csv
├── models/                 # created by train_model.py
│   ├── rf_model.pkl
│   ├── preprocessor.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   └── metrics.json
└── screenshots/            # add your app screenshots here
```

## Run Locally

```bash
git clone https://github.com/Dharshinirajesh38/Health-Risk-Classification.git
cd Health-Risk-Classification
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell:  .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open http://localhost:8501. The demo login is set in `.streamlit/secrets.toml`:

```toml
password = "choose-your-own-password"
```

## Screenshots

Add images from the `screenshots/` folder, for example:

```
![Dashboard](screenshots/dashboard.png)
![SHAP explanation](screenshots/shap.png)
```

## Tech Stack

Python · Streamlit · Pandas · NumPy · scikit-learn · SHAP · Plotly · Matplotlib · Joblib · ReportLab

## Future Work

- Validate on real, de-identified clinical data
- Probability calibration and per-class threshold tuning
- EHR integration and cloud deployment
- Automated tests and CI

## Author

**Dharshini** · [GitHub](https://github.com/Dharshinirajesh38) · [LinkedIn](https://linkedin.com/in/dharshini236)
