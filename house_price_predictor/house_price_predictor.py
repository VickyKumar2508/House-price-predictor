import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
import os
from sklearn.metrics import mean_absolute_error # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Load data
housing_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "train.csv"))

# Select target
y = housing_data.SalePrice

# Select improved features
housing_features = [
    "OverallQual", "GrLivArea", "GarageCars", 
    "FullBath", "YearBuilt",  "Fireplaces"
]
X = housing_data[housing_features]

# Train-test split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Train model
housing_model = RandomForestRegressor(n_estimators=100, random_state=1)
housing_model.fit(train_X, train_y)

# Validate model
val_predictions = housing_model.predict(val_X)
mae = mean_absolute_error(val_y, val_predictions)
print(f"\nModel validation complete. MAE: {mae:,.2f}\n")

# Feature importance visualization
importances = pd.Series(housing_model.feature_importances_, index=housing_features)
importances.sort_values().plot(kind='barh', figsize=(10,6), title="Feature Importances")
plt.tight_layout()
plt.show()

# Save the trained model
joblib.dump(housing_model, 'house_price_model.pkl')
print("Model saved as house_price_model.pkl")

# Ask user for input
print("\nEnter house details to predict price:")
inputs = {}
for feature in housing_features:
    dtype = int if 'Year' in feature or 'Cars' in feature or 'Fireplaces' in feature or 'Bath' in feature or 'Qual' in feature else float
    value = dtype(input(f"{feature}: "))
    inputs[feature] = value

# Prepare input and predict
user_data = pd.DataFrame([inputs])
prediction = housing_model.predict(user_data)
print(f"\nPredicted House Price: ${prediction[0]:,.2f}")
