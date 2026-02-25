
# ==============================================
# Healthcare Risk Classification - Training Script
# ==============================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==============================================
# 1️⃣ LOAD DATASET
# ==============================================

df = pd.read_csv("data/Healthcare_Risk_Classification_Dataset_Balanced.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)


# ==============================================
# 2️⃣ SEPARATE FEATURES AND TARGET
# ==============================================

# Remove Patient_ID (not useful for prediction)
X = df.drop(["Risk_Category", "Patient_ID"], axis=1)
y = df["Risk_Category"]

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)


# ==============================================
# 3️⃣ IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# ==============================================

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns


# ==============================================
# 4️⃣ CREATE PREPROCESSING PIPELINE
# ==============================================

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)


# ==============================================
# 5️⃣ CREATE MODEL PIPELINE (LOGISTIC REGRESSION)
# ==============================================

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


# ==============================================
# 6️⃣ TRAIN-TEST SPLIT
# ==============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==============================================
# 7️⃣ TRAIN MODEL
# ==============================================

model.fit(X_train, y_train)

print("Model Training Completed")


# ==============================================
# 8️⃣ MODEL EVALUATION
# ==============================================

# Test set prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)

print("\n========== MODEL PERFORMANCE ==========")
print("Test Accuracy:", accuracy)
print("Cross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))


# ==============================================
# 9️⃣ SAVE MODEL AND METRICS
# ==============================================

joblib.dump(model, "models/risk_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

joblib.dump(accuracy, "models/test_accuracy.pkl")
joblib.dump(cv_scores, "models/cv_scores.pkl")
joblib.dump(cm, "models/confusion_matrix.pkl")
joblib.dump(report, "models/classification_report.pkl")

print("\nModel and evaluation metrics saved successfully.")