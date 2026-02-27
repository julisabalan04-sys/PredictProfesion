import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and category mapping
model = joblib.load('modelo_bigfive.pkl')

category_mapping = {
    1: 'Actor/actriz',
    2: 'Cantante',
    3: 'Modelo',
    4: 'Tv, series',
    5: 'Radio',
    6: 'Tecnología',
    7: 'Deportes',
    8: 'Politica',
    9: 'Escritor'
}

# --- Streamlit App --- #
st.title('Predice tu profesión ideal')
nombre = st.text_input('Ingresa tu nombre')
st.write("Enter your Big Five personality scores to predict your potential profession.")

# Input fields for personality traits using sliders
# Using min, max, and mean from df.describe() as a guide
op_value = st.slider('Openness (op): Tendency to be imaginative, curious, and open to new experiences',
                     min_value=30.02, max_value=71.69, value=44.41, step=0.01)
co_value = st.slider('Conscientiousness (co): Tendency to be organized, reliable, and responsible',
                     min_value=7.85, max_value=49.63, value=22.97, step=0.01)
ex_value = st.slider('Extraversion (ex): Tendency to seek social stimulation and be outgoing',
                     min_value=18.69, max_value=59.82, value=40.76, step=0.01)
ag_value = st.slider('Agreeableness (ag): Tendency to be cooperative, compassionate, and empathetic',
                     min_value=9.30, max_value=40.58, value=22.91, step=0.01)
ne_value = st.slider('Neuroticism (ne): Tendency to experience negative emotions like anxiety, depression, and stress',
                     min_value=1.03, max_value=23.97, value=8.00, step=0.01)

st.write("### Your Entered Scores:")
st.write(f"Openness: {op_value}")
st.write(f"Conscientiousness: {co_value}")
st.write(f"Extraversion: {ex_value}")
st.write(f"Agreeableness: {ag_value}")
st.write(f"Neuroticism: {ne_value}")

# Create a button to trigger prediction
if st.button('Predict Profession'):
    # Prepare the input data for the model
    input_data = pd.DataFrame([{
        'op': op_value,
        'co': co_value,
        'ex': ex_value,
        'ag': ag_value,
        'ne': ne_value
    }])

    # Make prediction
    prediction_numeric = model.predict(input_data)[0]

    # Map the numerical prediction to the category name
    predicted_profession = category_mapping.get(prediction_numeric, 'Unknown Category')

    st.success(f"Hola **{nombre}** Based on your scores, your predicted profession is: **{predicted_profession}**")