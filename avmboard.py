# AVM with Geospatial Data + Google Maps Geocoding + Web Dashboard

import pandas as pd
import numpy as np
import googlemaps
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import streamlit as st
import folium
from streamlit_folium import st_folium

# Google Maps API Setup
gmaps = googlemaps.Client(key="AIzaSyAlCmkA_-4Cij0Gab4tU17Hi0kzl4P5U6g")

# Load Data
df = pd.read_csv("house_data_with_location.csv")  # Assumes columns: lat, long, plus home features

# Feature Engineering
df['age'] = 2025 - df['year_built']
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode', 'lat', 'long', 'age']
X = df[features]
y = df['price']

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

# Streamlit Web App
st.set_page_config(layout="wide")
st.title("Automated Real Estate Valuation Tool")
st.write(f"### Model MAE: ${mae:,.2f}")

# Input Form
st.sidebar.header("Property Info")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
sqft_living = st.sidebar.slider("Sqft Living", 300, 10000, 1800)
sqft_lot = st.sidebar.slider("Sqft Lot", 500, 20000, 5000)
floors = st.sidebar.slider("Floors", 1, 3, 1)
zipcode = st.sidebar.selectbox("Zipcode", sorted(df['zipcode'].unique()))
address = st.sidebar.text_input("Property Address", "1600 Amphitheatre Parkway, Mountain View, CA")
year_built = st.sidebar.number_input("Year Built", 1900, 2025, 2000)
age = 2025 - year_built

# Geocode the address
latitude, longitude = None, None
if address:
    try:
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            latitude = location['lat']
            longitude = location['lng']
        else:
            st.error("Address not found. Please check the input.")
    except Exception as e:
        st.error(f"Geocoding error: {e}")

if latitude and longitude:
    # Prediction
    new_input = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'zipcode': zipcode,
        'lat': latitude,
        'long': longitude,
        'age': age
    }])

    predicted_price = model.predict(new_input)[0]

    st.subheader("Predicted Property Price")
    st.success(f"${predicted_price:,.2f}")

    # Show Map
    m = folium.Map(location=[latitude, longitude], zoom_start=14)
    folium.Marker([latitude, longitude], popup=f"Predicted: ${predicted_price:,.0f}").add_to(m)
    st_folium(m, width=700, height=450)
else:
    st.warning("Enter a valid address to generate prediction and map.")
