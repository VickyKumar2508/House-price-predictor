# 🏠 House Price Predictor

A machine learning model that predicts house prices based on key features like overall quality, living area, number of garages, bathrooms, year built, and fireplaces. Built using Python and `scikit-learn`, trained on the [Kaggle House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## 🚀 Features

- Trained using Random Forest Regressor
- User-friendly input prompts for predictions
- Visualized feature importance
- Clean and well-commented code

## 📊 Model Performance

- **MAE (Mean Absolute Error)** on validation data: ~34,617

## 📦 Installation

```bash
git clone https://github.com/VickyKumar2508/house-price-predictor.git
cd house-price-predictor
pip install -r requirements.txt
python house_price_predictor.py
````



## 🧠 Features Used for Prediction

* OverallQual`: Overall material and finish quality
* GrLivArea`: Above grade (ground) living area square feet
* GarageCars`: Size of garage in car capacity
* FullBath`: Full bathrooms above grade
* YearBuilt`: Original construction date
* Fireplaces`: Number of fireplaces

## 📈 Example Output

Model validation complete. MAE: 34,617.80

Enter house details to predict price:
OverallQual: 7
GrLivArea: 2000
GarageCars: 2
FullBath: 2
YearBuilt: 2010
Fireplaces: 1

Predicted House Price: $231,764.17


