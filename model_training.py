import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Metrics
from sklearn.metrics import r2_score, mean_absolute_error

# Save Model
import joblib

# Load Dataset
df = pd.read_csv("C:\\Users\\Advance solution\\OneDrive\\Desktop\\New\\final_clean.csv")

# Features and Target
X = df.drop(columns=['Price'])
y = df['Price']

# Categorical Columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'
)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models Dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
}

# Store Results
results = {}

best_model = None
best_score = 0

# Train All Models
for name, model in models.items():

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Prediction
    y_pred = pipeline.predict(X_test)

    # Accuracy
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Store Results
    results[name] = {
        "R2 Score": r2,
        "MAE": mae
    }

    print(f"\n{name}")
    print("R2 Score:", r2)
    print("MAE:", mae)

    # Best Model Selection
    if r2 > best_score:
        best_score = r2
        best_model = pipeline

# Save Best Model
joblib.dump(best_model, "best_laptop_price_model.pkl")

print("\nBest Model Saved Successfully!")