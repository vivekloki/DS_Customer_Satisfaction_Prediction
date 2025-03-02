import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scripts.data_process import load_and_process_data

# ✅ Load the trained model
MODEL_FILE = "best_model.pkl"

def load_model():
    """ Load the saved model from .pkl file """
    try:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file not found! Train and save the model first.")
        return None

model = load_model()

# ✅ Streamlit UI
st.title("Customer Satisfaction Prediction 🚀")
st.markdown("Enter the details below to predict customer satisfaction.")

# ✅ Match feature names with trained model
feature_names = [
    "gender", "customer_type", "age", "type_of_travel", "class",
    "flight_distance", "inflight_wifi_service", "departure/arrival_time_convenient",
    "ease_of_online_booking", "gate_location", "food_and_drink",
    "online_boarding", "seat_comfort", "inflight_entertainment",
    "on-board_service", "leg_room_service", "baggage_handling", "checkin_service",
    "inflight_service", "cleanliness", "departure_delay_in_minutes", "arrival_delay_in_minutes"
]

# ✅ Create user input form
user_input = {}
for feature in feature_names:
    if feature in ["gender", "customer_type", "type_of_travel", "class"]:
        user_input[feature] = st.selectbox(
            f"{feature.replace('_', ' ').title()}",
            ["Male", "Female"] if feature == "gender" else ["Loyal", "Disloyal"]
        )
    else:
        user_input[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0, step=1.0)

if st.button("Predict Satisfaction"):
    if model:
        input_data = pd.DataFrame([user_input])
        input_data.columns = input_data.columns.str.lower().str.replace(" ", "_")

        # ✅ Pass `is_prediction=True` to avoid missing 'satisfaction' error
        processed_input = load_and_process_data(input_data, is_prediction=True)

        # ✅ Make Prediction
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0] if hasattr(model, "predict_proba") else None

        # ✅ Display Result
        st.subheader("Prediction Result:")
        st.write(f"**Predicted Satisfaction Level:** {'Satisfied' if prediction == 1 else 'Not Satisfied'}")

        if prediction_proba is not None:
            st.write(f"**Confidence:** {max(prediction_proba) * 100:.2f}%")
    else:
        st.error("No trained model found. Please train and log a model first.")