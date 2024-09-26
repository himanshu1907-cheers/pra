# prompt: create steramlit app for lasso regration


import streamlit as st
import pandas as pd
import pickle

# Load the trained Lasso Regression model
filename = 'lasso_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a Streamlit app
st.title("Monthly Revenue Prediction App")

# Input features from the user
st.header("Enter the following features:")
feature_names = ['average_order_value', 'customer_acquisition_cost', 'customer_lifetime_value',
                 'customer_churn_rate','social_media_engagement','average_shipping_time','customer_support_tickets']
user_input = {}
for feature_name in feature_names:
    user_input[feature_name] = st.number_input(feature_name, value=0.0)


# Create a button to make the prediction
if st.button("Predict Monthly Revenue"):
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_input])

    # Make the prediction using the loaded model
    predicted_revenue = loaded_model.predict(input_df)[0]

    # Display the prediction to the user
    st.header("Predicted Monthly Revenue:")
    st.write(f"{predicted_revenue:.2f}")