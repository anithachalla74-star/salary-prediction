"""
Salary Prediction — Full ML Pipeline
=====================================
Dataset  : 200 records (YearsExperience, TestScore, Education → Salary)
Models   : 4 Linear Regression variants
Best Model saved to: best_model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ── 1. Load & Explore ─────────────────────────────────────────────────────────
df = pd.read_csv('salary_data.csv')
print("Shape:", df.shape)
print(df.describe())
print("\nEducation value counts:\n", df['Education'].value_counts())

# ── 2. Train / Test Split ─────────────────────────────────────────────────────
X = df.drop('Salary', axis=1)
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'─'*40}")
    print(f"  Model : {name}")
    print(f"  RMSE  : ${rmse:,.2f}")
    print(f"  R²    : {r2:.4f}")
    return {'rmse': round(rmse, 2), 'r2': round(r2, 4)}

# ── 3. Model 1 — Single Feature: Experience ───────────────────────────────────
m1 = LinearRegression()
m1.fit(X_train[['YearsExperience']], y_train)
r1 = evaluate("Single — YearsExperience", y_test, m1.predict(X_test[['YearsExperience']]))

# ── 4. Model 2 — Single Feature: TestScore ────────────────────────────────────
m2 = LinearRegression()
m2.fit(X_train[['TestScore']], y_train)
r2_ = evaluate("Single — TestScore", y_test, m2.predict(X_test[['TestScore']]))

# ── 5. Model 3 — Multiple Features (numeric only) ─────────────────────────────
m3 = LinearRegression()
m3.fit(X_train[['YearsExperience', 'TestScore']], y_train)
r3 = evaluate("Multi — Experience + TestScore", y_test,
               m3.predict(X_test[['YearsExperience', 'TestScore']]))

# ── 6. Model 4 — All Features + One-Hot Encoding ──────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first'), ['Education'])
], remainder='passthrough')

best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor',    LinearRegression())
])
best_pipeline.fit(X_train, y_train)
r4 = evaluate("Multi — All Features (OHE Education)", y_test,
               best_pipeline.predict(X_test))

# ── 7. Save Best Model ─────────────────────────────────────────────────────────
joblib.dump(best_pipeline, 'best_model.pkl')
print("\n✅ Best model saved → best_model.pkl")

# ── 8. Summary Table ──────────────────────────────────────────────────────────
print("\n" + "═"*55)
print("  MODEL COMPARISON SUMMARY")
print("═"*55)
summary = pd.DataFrame({
    'Model': ['Single (Experience)', 'Single (TestScore)',
              'Multi (No Cat)', 'All Features (Best)'],
    'RMSE ($)': [r1['rmse'], r2_['rmse'], r3['rmse'], r4['rmse']],
    'R²':       [r1['r2'],   r2_['r2'],   r3['r2'],   r4['r2']]
})
summary['Winner'] = summary['R²'] == summary['R²'].max()
print(summary.to_string(index=False))

# ── 9. Quick Demo Prediction ──────────────────────────────────────────────────
sample = pd.DataFrame({'YearsExperience': [5.0], 'TestScore': [78.0], 'Education': ['Bachelor']})
pred = best_pipeline.predict(sample)[0]
print(f"\n📌 Sample prediction: 5 yrs exp, 78 score, Bachelor → ${pred:,.2f}")
