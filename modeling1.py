# modeling1.py
# 🧠 Predicting the number of threatened species (Regression)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 📥 Load the dataset
df = pd.read_csv("/Users/ayushipatel/Desktop/adv/ThreatenedSpecies/data/cleaned_ThreatenedSpecies.csv")

# 🧹 Drop unnecessary columns
df = df.drop(columns=["Source"])

# 🎯 Define features and target
X = df[["Country", "Year", "Species_Type"]]
y = df["Count"]

# 🔤 Define which features are categorical for encoding
categorical_features = ["Country", "Species_Type"]

# ⚙️ Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"  # Keep numerical 'Year'
)

# 🌲 Create a Random Forest Regressor pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# ✂️ Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🏋️ Train the model
model.fit(X_train, y_train)

# 📊 Make predictions
y_pred = model.predict(X_test)

# 🧮 Evaluate the model
print("📈 Model Evaluation:")
print("Root Mean Squared Error (RMSE):", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))




