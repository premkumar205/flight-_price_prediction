import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Flight Price Predictor", layout="wide")

st.title("✈️ Flight Price Prediction App")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("flight_price_prediction_model.pkl")

# Load dataset for encoders
@st.cache_resource
def load_data():
    return pd.read_excel("Data_Train.xlsx")

try:
    model = load_model()
    df = load_data()

    st.sidebar.header("🛫 Enter Flight Details")

    # ----------- USER INPUTS -----------

    airline = st.selectbox("Airline", df["Airline"].unique())
    source = st.selectbox("Source", df["Source"].unique())
    destination = st.selectbox("Destination", df["Destination"].unique())
    additional_info = st.selectbox("Additional Info", df["Additional_Info"].unique())

    total_stops = st.selectbox("Total Stops", df["Total_Stops"].unique())

    duration = st.number_input("Duration (in minutes)", min_value=0, value=120)

    day = st.slider("Day", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
    year = st.slider("Year", 2018, 2025, 2019)

    dep_hour = st.slider("Departure Hour", 0, 23, 10)
    dep_min = st.slider("Departure Minute", 0, 59, 30)

    arr_hour = st.slider("Arrival Hour", 0, 23, 12)
    arr_min = st.slider("Arrival Minute", 0, 59, 45)

    route1 = st.text_input("Route 1")
    route2 = st.text_input("Route 2")
    route3 = st.text_input("Route 3")
    route4 = st.text_input("Route 4")
    route5 = st.text_input("Route 5")

    # ----------- ENCODING -----------

    def encode_column(col_name, value):
        le = LabelEncoder()
        le.fit(df[col_name])
        return le.transform([value])[0]

    # ----------- PREDICTION -----------

    if st.button("💰 Predict Flight Price"):

        input_dict = {
            "Airline": encode_column("Airline", airline),
            "Source": encode_column("Source", source),
            "Destination": encode_column("Destination", destination),
            "Duration": duration,
            "Total_Stops": encode_column("Total_Stops", total_stops),
            "Additional_Info": encode_column("Additional_Info", additional_info),
            "Day": day,
            "Month": month,
            "Year": year,
            "Dep_hour": dep_hour,
            "Dep_min": dep_min,
            "Arr_hour": arr_hour,
            "Arr_min": arr_min,
            "route1": route1,
            "route2": route2,
            "route3": route3,
            "route4": route4,
            "route5": route5
        }

        input_df = pd.DataFrame([input_dict])

        try:
            prediction = model.predict(input_df)[0]

            st.markdown("---")
            st.subheader("📊 Prediction Result")
            st.success(f"💰 Estimated Flight Price: ₹ {int(prediction)}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

except Exception as e:
    st.error(f"❌ Error Loading Model or Data: {e}")