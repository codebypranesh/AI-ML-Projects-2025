import streamlit as st
import joblib
import pandas as pd
import os

st.title("🎬 Netflix User Rating Predictor")

# Load model
model = joblib.load('netflix_rating_model.pkl')
st.success("✅ Model loaded successfully!")

# User inputs
type_input = st.selectbox("Select Type", ['Movie', 'TV Show'])
release_year = st.slider("Release Year", 1940, 2025, 2020)
duration = st.number_input("Duration (in minutes)", min_value=1, max_value=500, value=90)

# Genre checkboxes
genres = ['Action & Adventure', 'Anime Features', 'Comedies', 'Crime TV Shows', 'Documentaries']
genre_features = {}
for genre in genres:
    genre_features[f'genre_{genre}'] = st.checkbox(genre)

# Prepare input data
input_data = {
    'type': 0 if type_input == 'Movie' else 1,
    'release_year': release_year,
    'duration': duration,
}
input_data.update(genre_features)

# Fill in missing features (if model expects more than selected)
for col in model.feature_names_in_:
    if col not in input_data:
        input_data[col] = 0  # default value for missing columns

# Create DataFrame with correct column order
input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Show the input data
st.write("🧾 Input Data Preview:")
st.write(input_df)

# Predict on button click
if st.button("Predict Rating"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Predicted User Rating: ⭐ {round(prediction, 1)} / 10")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
