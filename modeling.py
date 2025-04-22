import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn tools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import  r2_score, mean_absolute_error

# XGBoost model
from xgboost import XGBRegressor

# Load and Prepare Data
df = pd.read_csv("data/cleaned_ThreatenedSpecies.csv").drop(columns=["Source"])
df["Decade"] = (df["Year"] // 10) * 10  # Add 'Decade' feature

X = df[["Country", "Species_Type", "Year", "Decade"]]
y = df["Count"]

#  Preprocessing
categorical_features = ["Country", "Species_Type", "Decade"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")

# XGBoost Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(objective="reg:squarederror", random_state=42))
])

#  Grid Search Parameters
param_grid = {
    "regressor__n_estimators": [100, 300],
    "regressor__learning_rate": [0.05, 0.1],
    "regressor__max_depth": [3, 5],
    "regressor__subsample": [0.8, 1.0],
    "regressor__colsample_bytree": [0.8, 1.0],
}

#  Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training with GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Best Model & Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Final Optimized XGBoost Model")
print(f"Best  Parameters: {grid_search.best_params_}")
print(f"MAE  : {mae:.2f}")
print(f"R²  : {r2:.3f}")

#  Feature Importance
feature_names = best_model.named_steps["preprocessor"].transformers_[0][1].get_feature_names_out(categorical_features)
full_feature_names = np.concatenate([feature_names, ["Year"]])
importances = best_model.named_steps["regressor"].feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(15), importances[indices][:15])
plt.yticks(range(15), full_feature_names[indices][:15])
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances ")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Count")
plt.ylabel("Predicted Count")
plt.title(" Actual vs Predicted Count")
plt.tight_layout()
plt.show()
