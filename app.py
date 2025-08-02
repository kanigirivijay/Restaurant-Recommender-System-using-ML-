import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. First, we load our saved "brain" of the app ---
try:
    the_model = joblib.load('recommender_model.joblib')
    the_imputer = joblib.load('imputer.joblib')
except FileNotFoundError:
    st.error("Error: The model files are missing. Please make sure they are in the same folder as this app.")
    st.stop() 

# --- 2. Next, we set up how our app looks ---
st.set_page_config(
    page_title="My First Recommender App! 🍔",
    page_icon="🍔",
    layout="wide"
)
st.title("My First Recommender App! 🍔") 
st.markdown("I made this app to predict if a customer will order from a restaurant.")

# We load the vendors file just to show the vendor IDs
try:
    vendors_df = pd.read_csv('vendors.csv')
    vendors_df.rename(columns={'id': 'vendor_id'}, inplace=True)
except FileNotFoundError:
    st.error("Error: vendors.csv not found. Please make sure it's in the same folder.")
    st.stop()


# This creates a sidebar on the left side of the app
st.sidebar.header('Change the Restaurant Details:')

# --- 3. Now all the features are selectable by the user for the demo! ---
# We will still show the vendor ID, but the other features are manual inputs
vendor_ids = vendors_df['vendor_id'].unique()
selected_vendor_id = st.sidebar.selectbox('Select a Restaurant (by ID):', vendor_ids)

# These are the manual inputs for the demo
user_distance = st.sidebar.slider('Distance (km)', min_value=0.5, max_value=20.0, value=5.0)
user_rating = st.sidebar.slider('Vendor Rating', min_value=0.0, max_value=5.0, value=4.5, step=0.1)
user_delivery_charge = st.sidebar.slider('Delivery Charge', min_value=0.0, max_value=20.0, value=3.0)
user_is_open = st.sidebar.radio('Is the restaurant open?', ['Yes', 'No'])

# --- 4. The magic happens here when you click the button ---
if st.sidebar.button('Make a Prediction!'):
    # The app takes the values from the sliders and buttons
    user_input = pd.DataFrame({
        'distance_km': [user_distance],
        'delivery_charge': [user_delivery_charge],
        'serving_distance': [np.nan],
        'is_open': [1 if user_is_open == 'Yes' else 0],
        'commission': [np.nan],
        'discount_percentage': [np.nan],
        'vendor_rating': [user_rating],
        'prepration_time': [np.nan],
        'rank': [np.nan],
        'one_click_vendor': [np.nan],
        'country_id': [np.nan],
        'city_id': [np.nan],
        'vendor_category_id': [np.nan]
    })
    
    # It prepares the data and uses the model's "brain" to make a guess
    user_input_filled = the_imputer.transform(user_input)
    prediction_result = the_model.predict(user_input_filled)
    
    # We show the result to the user!
    st.subheader(f"Prediction for Vendor ID: **{selected_vendor_id}**")
    
    # It checks if the prediction is a '1' or a '0' and shows the correct message
    if prediction_result[0] == 1:
        st.success('A customer WILL likely order from this restaurant! 🎉')
    else:
        st.error('A customer will NOT likely order from this restaurant.')