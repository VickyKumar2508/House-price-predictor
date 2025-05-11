from house_price_predictor import predict_price
import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load('house_price_model.pkl')

# Set page config
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Predictor")

# Input fields
st.write("Enter the details of the house:")

overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, value=1500)
garage_cars = st.slider("Garage Capacity (cars)", 0, 5, 1)
full_bath = st.slider("Number of Full Bathrooms", 0, 5, 1)
year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
fireplaces = st.slider("Number of Fireplaces", 0, 5, 1)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "FullBath": full_bath,
        "YearBuilt": year_built,
        "Fireplaces": fireplaces
    }])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")

def main():
    st.write("Run the app by interacting with the inputs above.")
    
if __name__ == "__main__":
    main()

