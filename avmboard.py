import streamlit as st
import pandas as pd
import os
import google.auth
from google.cloud import storage
from datetime import datetime

# Load AVM Data
@st.cache_data
def load_data():
    return pd.read_csv("house_data_with_location.csv")

data = load_data()
st.title("🏠 Automated Valuation Model (AVM) Dashboard")
st.map(data[['latitude', 'longitude']])

# Registration Form
st.header("📝 Register for Updates")
name = st.text_input("Full Name")
email = st.text_input("Email Address")
city = st.text_input("City")
age = st.number_input("Age", min_value=18, max_value=100, step=1)

if st.button("Submit Registration"):
    if name and email and city:
        reg_data = pd.DataFrame({
            "name": [name],
            "email": [email],
            "city": [city],
            "age": [age],
            "timestamp": [datetime.now().isoformat()]
        })
        csv_filename = "registration_data.csv"
        reg_data.to_csv(csv_filename, index=False)

        try:
            from google.oauth2 import service_account
            import json

            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )

            client = storage.Client(credentials=credentials, project=credentials.project_id)
            bucket_name = "registration_data_realstate"
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(csv_filename)
            blob.upload_from_filename(csv_filename)

            st.success("✅ Registration submitted and uploaded to GCS successfully!")
        except Exception as e:
            st.error(f"❌ Failed to upload CSV to GCS: {e}")
    else:
        st.warning("Please fill in all the required fields.")
