# AVM with Geospatial Data + Google Maps Geocoding + Google Maps Embed + Web Dashboard + Registration Form to CSV and GCS

import pandas as pd
import numpy as np
import googlemaps
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import streamlit as st
from google.cloud import storage
import json
import os

# Google Maps API Setup
gmaps = googlemaps.Client(key="AIzaSyAlCmkA_-4Cij0Gab4tU17Hi0kzl4P5U6g")

# GCS Bucket Name
GCS_BUCKET_NAME = "registration_data_realstate"

# Function to upload file to GCS
def upload_csv_to_gcs(local_file_path, bucket_name, destination_blob_name):
    try:

        # Specify your Google Cloud Project ID
        project_id = "skilful-sensor-457808-v8"  # Replace with your actual project ID
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        st.success("CSV uploaded to GCS successfully.")
    except Exception as e:
        st.error(f"Failed to upload CSV to GCS: {e}")

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

# Registration Form
st.sidebar.header("Register to Access Full Dashboard")
with st.sidebar.form("registration_form"):
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    city = st.text_input("City")
    purpose = st.selectbox("Purpose", ["Buying", "Selling", "Research", "Other"])
    submitted = st.form_submit_button("Register")

if submitted:
    user_data = {
        "name": name,
        "email": email,
        "phone": phone,
        "city": city,
        "purpose": purpose,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    df_user = pd.DataFrame([user_data])
    local_csv = f"temp_registration_{email.replace('@', '_')}.csv"
    df_user.to_csv(local_csv, index=False)
    upload_csv_to_gcs(local_csv, GCS_BUCKET_NAME, f"registrations/{local_csv}")
    os.remove(local_csv)

# Input Form for AVM
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

    # Show Google Map instead of folium
    map_url = f"https://www.google.com/maps/embed/v1/place?key=AIzaSyAlCmkA_-4Cij0Gab4tU17Hi0kzl4P5U6g&q={latitude},{longitude}"
    st.markdown(f"""
    <iframe
        width="100%"
        height="450"
        style="border:0;"
        loading="lazy"
        allowfullscreen
        src="{map_url}">
    </iframe>
    """, unsafe_allow_html=True)
else:
    st.warning("Enter a valid address to generate prediction and map.")
