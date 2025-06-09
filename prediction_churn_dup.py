import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import keras # Import keras directly for load_model

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

## Define the base path for your assets (model, encoders, scaler)
## BASE_PATH = '/Users/puneetch/Desktop/Python/basics/' # Make sure this path is correct

## Load the model
model = keras.models.load_model('/Users/puneetch/Desktop/Python/basics/model.h5')

## Load the encoders and scalar
with open('/Users/puneetch/Desktop/Python/basics/label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('/Users/puneetch/Desktop/Python/basics/onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('/Users/puneetch/Desktop/Python/basics/scaler.pkl','rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title("Customer Churn Prediction")

# Ensure the options in selectbox match the categories your encoders were trained on
# For gender, it would typically be ['Male', 'Female']
gender = st.selectbox('Gender', label_encoder_gender.classes_.tolist()) # Convert numpy array to list for selectbox
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0].tolist()) # Convert numpy array to list

age = st.slider('Age', 18, 100)
balance = st.number_input('Balance', value=0.0) # Add default value for number_input
credit_score = st.number_input('Credit Score', value=600) # Add default value
estimated_salary = st.number_input('Estimated Salary', value=0.0) # Add default value
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4) # Typically 1-4 products
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


## Prepare the input Data (as a dictionary, then to DataFrame)
input_data = {
    'CreditScore' : credit_score,
    'Geography' : geography,
    'Gender' : gender, # THIS IS THE CRITICAL CHANGE: Use the selected 'gender' variable directly
    'Age' : age,
    'Tenure' : tenure,
    'Balance' : balance,
    'NumOfProducts' : num_of_products,
    'HasCrCard' : has_credit_card,
    'IsActiveMember' : is_active_member,
    'EstimatedSalary' : estimated_salary
}

# Convert the dictionary to a Pandas DataFrame
input_df = pd.DataFrame([input_data])


# --- Transformation Logic ---

# 1. Label Encode Gender
# Ensure the gender column is treated as a categorical input for the encoder
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

# 2. One-Hot Encode Geography
# The onehot_encoder_geo.transform expects a 2D array, e.g., [[value]]
# And the column names need to match the original training data
geo_encoded = onehot_encoder_geo.transform(input_df[['Geography']])
geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


# 3. Drop original categorical columns and concatenate
input_df = input_df.drop(columns=['Geography'], axis=1) # Drop original Geography column

# Concatenate the transformed geography features
# Make sure the order of concatenation respects the order used during model training
# For example, if you trained with ['CreditScore', 'Gender', 'Age', ..., 'Geography_France', 'Geography_Germany']
# you need to recreate that order.
# A robust way is to define all expected columns and reindex.

# Example of robust column ordering:
# Assuming your original training features (excluding original Gender and Geography) were:
## numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
# And your one-hot encoded geo columns are what onehot_encoder_geo.get_feature_names_out() returns.
# And gender is now numerical.

# Re-assemble the DataFrame with the correct column order
# This is crucial for the model to make correct predictions
final_input_df = pd.concat([input_df, geo_df], axis=1)


# Ensure the final columns match the exact order and names expected by the model
# It's a good practice to have a list of feature names from your training data, e.g.,
# feature_columns_at_training = ['CreditScore', 'Gender', 'Age', ..., 'Geography_France', 'Geography_Germany', ...]
# If you have that, you can do: final_input_df = final_input_df[feature_columns_at_training]
# If not, ensure the order is correct based on how you preprocessed your training data.


## Scaling the concatenated data
# The scaler expects numerical data, so ensure all columns in final_input_df are numerical.
scaled_input = scaler.transform(final_input_df)


## Prediction
prediction = model.predict(scaled_input)


## Prediction probability
prediction_probability = prediction[0][0]

st.write("### Prediction Result:",prediction)
if prediction_probability > 0.5:
    st.write("The customer is likely to exit.")
else:
    st.write("The customer will stay.")
