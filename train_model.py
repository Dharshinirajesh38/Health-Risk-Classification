# ==============================================
# Healthcare Risk Classification - Training Script
# Trains Logistic Regression (baseline) and Random Forest,
# saves the Random Forest artifacts that app.py loads.
# Run from the project root:  python train_model.py
# ==============================================

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA_PATH = "data/Healthcare_Risk_Classification_Dataset_Balanced.csv"
MODEL_DIR = "models"
SEED = 42
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------- 1. Load ----------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["Risk_Category"] = df["Risk_Category"].astype(str).str.strip().str.upper()
print("Loaded:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()),
      "| Duplicate rows:", int(df.duplicated().sum()))
print(df["Risk_Category"].value_counts().to_string())


# ---------- 2. Feature engineering (same formulas as app.py) ----------
def add_engineered_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["MAP"] = ((d["Systolic_BP"] + 2 * d["Diastolic_BP"]) / 3).round(2)
    d["BMI_Age_Index"] = (d["BMI"] * d["Age"] / 100).round(3)
    d["Metabolic_Risk_Score"] = (d["HbA1c_percent"] * 0.6 + d["BMI"] * 0.4).round(3)
    d["Activity_Score"] = (d["Physical_Activity_hours_per_week"] * 200 + d["Daily_Steps"]).round(0)
    d["Sleep_Quality"] = np.where(d["Sleep_Hours_per_night"].between(7, 9), 1, 0)
    return d


df = add_engineered_features(df)


# ---------- 3. Features / target ----------
X = df.drop(columns=[c for c in ["Risk_Category", "Patient_ID"] if c in df.columns])
le = LabelEncoder()
y = le.fit_transform(df["Risk_Category"])

num_feats = X.select_dtypes(include=["number"]).columns.tolist()
cat_feats = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
print("Numeric:", len(num_feats), "| Categorical:", cat_feats)


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(drop="first", sparse_output=False,
                              handle_unknown="ignore"), cat_feats),
    ])


# ---------- 4. Split (preprocessor is fit on TRAIN only -> no leakage) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=SEED, n_jobs=-1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = {}
print("\n========== MODEL COMPARISON ==========")
for name, clf in models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
    }
    print(f"{name:20s} {results[name]}")


# ---------- 5. Final model = Random Forest (what app.py + SHAP use) ----------
preprocessor = make_preprocessor()
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

best_name = max(results, key=lambda n: results[n]["cv_mean_accuracy"])
print("\nFinal model (best CV accuracy):", best_name)
rf = models[best_name]
rf.fit(X_train_p, y_train)
y_pred = rf.predict(X_test_p)

cat_out = (preprocessor.named_transformers_["cat"]
           .get_feature_names_out(cat_feats).tolist() if cat_feats else [])
feature_names = num_feats + cat_out

print(f"\n========== {best_name.upper()} - TEST REPORT ==========")
print(classification_report(y_test, y_pred, target_names=le.classes_))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix (rows=true, cols=pred), classes:", list(le.classes_))
print(cm)

imp = getattr(rf, "feature_importances_", None)
if imp is None:
    imp = np.abs(rf.coef_).mean(axis=0)
top = sorted(zip(feature_names, imp), key=lambda t: -t[1])[:8]
print("\nTop features:", [(n, round(float(v), 3)) for n, v in top])


# ---------- 6. Save (file names match app.py) ----------
rf.background_ = X_train_p[:100]          # used by SHAP LinearExplainer in app.py
joblib.dump(rf, f"{MODEL_DIR}/rf_model.pkl")   # file name kept so app.py needs no change
joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.pkl")
joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")
joblib.dump(feature_names, f"{MODEL_DIR}/feature_names.pkl")

with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump({
        "comparison": results,
        "final_model": best_name,
        "classes": list(le.classes_),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=le.classes_, output_dict=True),
    }, f, indent=2)

print("\nSaved: rf_model.pkl, preprocessor.pkl, label_encoder.pkl, "
      "feature_names.pkl, metrics.json ->", MODEL_DIR)
