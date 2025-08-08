# app.py
import numpy as np
import pandas as pd
import sklearn
import joblib
import os, pickle
import streamlit as st
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('ParisHousing.csv')

X = df.drop("price", axis=1)
y = df["price"]

# Assuming X and y are already defined as features and target variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Page configuration

page_title="Melbourne House Price Predictor",
page_icon="🏠",
layout="centered",
initial_sidebar_state="auto",

base = Path.cwd()
scaler_fp = Path('models/scaler.pkl')
#model_fp  = Path('models/rf_model.pkl')

model_fp = RandomForestRegressor()
model_fp.fit(X_train, y_train)

path = Path('models/')

if not path.exists() or not path.exists():
    st.error("Model artifacts not found.\n")
    st.stop()

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X) 

def main():
    st.title("🏠 Melbourne House Price Predictor")
    st.write("Enter your property details in the sidebar and click **Predict**.")

    # Sidebar Inputs
    st.sidebar.header("Property Features")
    square_m   = st.sidebar.number_input("Living Area (sqm)",  10, 10000, 50)
    rooms      = st.sidebar.number_input("Number of Rooms",     1,    20,  3)
    has_yard   = st.sidebar.checkbox("Yard")
    has_pool   = st.sidebar.checkbox("Pool")
    floors     = st.sidebar.number_input("Floors",              1,    10,  1)
    city_code  = st.sidebar.number_input("City Code",           1,  99999, 50000)
    part_rng   = st.sidebar.slider("Neighborhood Range (0–10)", 0, 10, 5)
    prev_own   = st.sidebar.number_input("Previous Owners",     0,    10,  1)
    year_built = st.sidebar.number_input("Year Built",        1900,  2025, 2005)
    is_new     = st.sidebar.checkbox("New / Renovated")
    has_storm  = st.sidebar.checkbox("Storm Protector")
    basement   = st.sidebar.number_input("Basement (sqm)",      0,  10000,   0)
    attic      = st.sidebar.number_input("Attic (sqm)",         0,  10000,   0)
    garage     = st.sidebar.number_input("Garage (sqm)",        0,   2000,   0)
    has_store  = st.sidebar.checkbox("Storage Room")
    has_guest  = st.sidebar.checkbox("Guest Room")

    # Predict button
    if st.button("Predict Price"):
        x = np.array([[ 
            square_m,
            rooms,
            int(has_yard),
            int(has_pool),
            floors,
            city_code,
            part_rng,
            prev_own,
            year_built,
            int(is_new),
            int(has_storm),
            basement,
            attic,
            garage,
            int(has_store),
            int(has_guest)
        ]])

        # Scale & predict
        x_scaled = scaler.transform(x)
        price    = model_fp.predict(x_scaled)[0]

        st.success(f"🎉 Estimated Price: ${price:,.2f}")

if __name__ == "__main__":
    main()

