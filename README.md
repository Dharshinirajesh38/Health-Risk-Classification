# 🏥 Healthcare Risk Classification System

A machine-learning web app that predicts a patient's health risk level (**LOW / MEDIUM / HIGH / CRITICAL**) from clinical and lifestyle data, and explains every prediction with SHAP.

> ⚠️ **Educational project, not a medical device.** It is trained on a synthetic dataset and must not be used for real diagnosis or treatment decisions. The follow-up suggestions are simple rule-based hints.

## Features

- **Dashboard:** upload a CSV/Excel file, see the cleaning log, population KPIs, risk distribution, Age vs BMI, metabolic-risk and smoking charts, and model performance
- **Patient DB:** pick a patient ID, run the AI diagnostic, compare it with the historical label, and download a PDF report
- **Assessment:** enter a new patient manually and get the predicted risk, SHAP explanation and a PDF report. A live table shows **normal / borderline / abnormal** ranges for each value, adjusted for the selected gender
- **Auto data cleaning:** column renaming, label standardisation, type coercion, missing-value imputation, duplicate removal, outlier clipping (1.5×IQR)
- **Feature engineering:** MAP, BMI-Age index, metabolic risk score, activity score, sleep quality
- **Explainable AI:** SHAP shows which factors push the risk **higher** (orange) or **lower** (green)
- **Login and page navigation** with a *Next* button between pages

## Model

| Step | Detail |
|---|---|
| Data | 1,000 synthetic patients, 15 clinical/lifestyle features, 250 per class |
| Split | 80/20 stratified, `random_state=42` |
| Preprocessing | StandardScaler (numeric) + OneHotEncoder (categorical), fit on training data only |
| Models compared | Logistic Regression vs Random Forest (200 trees, balanced class weights) |
| Selection | Best 5-fold cross-validation accuracy is saved automatically |
| Explainability | SHAP `LinearExplainer`; impact is measured on risk severity (LOW=0 … CRITICAL=3) |

### Results

| Model | Test accuracy | Macro F1 | CV accuracy (mean ± std) |
|---|---|---|---|
| **Logistic Regression (final)** | **0.95** | **0.95** | **0.933 ± 0.025** |
| Random Forest | 0.66 | 0.65 | 0.660 ± 0.014 |

Logistic Regression clearly beat Random Forest, which suggests the synthetic labels follow a mostly linear rule of the inputs. Full metrics and the confusion matrix are saved in `models/metrics.json` and shown in the app.

### Limitations

- The dataset is **synthetic and perfectly balanced**, so these results will not carry over to real patients.
- `Activity_Score` and `Metabolic_Risk_Score` use assumed formulas, not clinical standards. `MAP` is a linear mix of the two BP columns, so SHAP can split their effect arbitrarily (multicollinearity).
- Labels may not follow real clinical definitions, and there is no external validation.
- In a clinical setting, predicting LOW for a CRITICAL patient is the costly error, so per-class recall matters more than overall accuracy.

## Input Columns

`Patient_ID`, `Age`, `Gender`, `BMI`, `Systolic_BP`, `Diastolic_BP`, `Cholesterol_mg_dL`, `HbA1c_percent`, `Smoking_Status`, `Alcohol_Consumption_per_week`, `Physical_Activity_hours_per_week`, `Sleep_Hours_per_night`, `Avg_Heart_Rate`, `Daily_Steps`, `Family_History_Diabetes`, `Family_History_Heart_Disease`, and the target `Risk_Category`.

Uploaded files with similar names (for example `Sex`, `Systolic`, `HbA1c`, `Steps`, `Risk`) are renamed automatically.

## Reference Ranges Used in the Assessment Page

General adult guidelines (WHO, ACC/AHA, ADA, ATP III, NIAAA). Most are the same for men and women; **alcohol limits differ** (men ≤ 14 drinks/week, women ≤ 7).

| Parameter | Normal | Borderline | Abnormal |
|---|---|---|---|
| BMI | 18.5 – 24.9 | 25 – 29.9 | < 18.5 or ≥ 30 |
| Blood pressure (mmHg) | < 120 / < 80 | 120–139 / 80–89 | ≥ 140 / ≥ 90 |
| Cholesterol (mg/dL) | < 200 | 200 – 239 | ≥ 240 |
| HbA1c (%) | < 5.7 | 5.7 – 6.4 | ≥ 6.5 |
| Heart rate (bpm) | 60 – 100 | 50–59 or 101–110 | < 50 or > 110 |
| Sleep (hrs/night) | 7 – 9 | 6–6.9 or > 9 | < 6 |
| Activity (hrs/week) | ≥ 2.5 | 1 – 2.4 | < 1 |
| Daily steps | ≥ 7,000 | 5,000 – 6,999 | < 5,000 |

## Project Structure

```
Health-Risk-Classification/
├── app.py                  # Streamlit application
├── train_model.py          # Trains, compares and saves the model
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── Healthcare_Risk_Classification_Dataset_Balanced.csv
├── models/                 # created by train_model.py
│   ├── rf_model.pkl        # saved best model (file name kept for app.py)
│   ├── preprocessor.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   └── metrics.json
├── screenshots/            # app screenshots
├── Report_Health_Risk_Classification.pdf
└── patient_report.pdf      # sample generated report
```

## Run Locally

Python 3.10 or newer (tested on 3.13).

```powershell
git clone https://github.com/Dharshinirajesh38/Health-Risk-Classification.git
cd Health-Risk-Classification
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Create `.streamlit/secrets.toml` with your own demo password (this file is git-ignored):

```toml
password = "choose-your-own-password"
```

Open http://localhost:8501 and log in with username `admin` and that password.

## Screenshots

<!-- After adding images to the screenshots folder, remove the comment markers below.
![Dashboard](screenshots/dashboard.png)
![Patient diagnostic with SHAP](screenshots/patient_db.png)
![Clinical assessment](screenshots/assessment.png)
-->

## Tech Stack

Python · Streamlit · Pandas · NumPy · scikit-learn · SHAP · Plotly · Matplotlib · Joblib · ReportLab

## Future Work

- Validate on real, de-identified clinical data
- Probability calibration and per-class threshold tuning
- EHR integration and cloud deployment
- Automated tests and CI

## Author

**Dharshini** 
