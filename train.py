import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── Load data ───────────────────────────────────────
df = pd.read_csv("data/diabetes.csv")

# ── Clean data ──────────────────────────────────────
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols] = df[cols].replace(0, np.nan)
for col in cols:
    df[col].fillna(df[col].median(), inplace=True)

# ── Feature Engineering ─────────────────────────────
df['BMI_Age']          = df['BMI'] * df['Age']
df['Glucose_Insulin']  = df['Glucose'] * df['Insulin']
df['Glucose_BMI']      = df['Glucose'] * df['BMI']

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ── Balance ─────────────────────────────────────────
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

# ── Split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale ────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Models ───────────────────────────────────────────
print("Training models...\n")

models = {
    "LogisticRegression":  LogisticRegression(max_iter=1000, C=0.1),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
    "RandomForest":        RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "GradientBoosting":    GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "SVM":                 SVC(kernel='rbf', C=10, gamma='scale', probability=True),
    "XGBoost":             XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, eval_metric='logloss', verbosity=0),
}

results = {}
trained = {}

for name, m in models.items():
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    results[name] = acc
    trained[name] = m
    print(f"{name}: {acc*100:.2f}%")

# ── Ensemble of top 3 ────────────────────────────────
top3 = sorted(results, key=results.get, reverse=True)[:3]
print(f"\nBuilding ensemble from top 3: {top3}")

ensemble = VotingClassifier(
    estimators=[(n, trained[n]) for n in top3],
    voting='soft'
)
ensemble.fit(X_train, y_train)
ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
print(f"Ensemble Accuracy: {ens_acc*100:.2f}%")

results['Ensemble'] = ens_acc
trained['Ensemble'] = ensemble

# ── Save best ────────────────────────────────────────
best_name = max(results, key=results.get)
joblib.dump(trained[best_name], "model/diabetes_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print(f"\n✅ Best Model: {best_name}")
print(f"✅ Best Accuracy: {results[best_name]*100:.2f}%")
print("✅ Saved to model/diabetes_model.pkl")