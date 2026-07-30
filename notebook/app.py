import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

file_paths = os.path.dirname(os.path.abspath(__file__))

# load artifacts
with open(os.path.join(file_paths, 'model_artifacts.pkl'), 'rb') as f:
    artifacts = pickle.load(f)

feature_matrix = artifacts['feature_matrix']
feature_columns = artifacts['feature_columns']
weights_adjusted = artifacts['weights_adjusted']
scaler = artifacts['scaler']

listings_rec = pd.read_csv(os.path.join(SCRIPT_DIR, 'listings_rec.csv'))

# recommender function
def recommend_hybrid(user_prefs, alpha=0.7, top_n=5):
    query = pd.DataFrame(0, index=[0], columns=feature_columns)
    for key, value in user_prefs.items():
        if key in query.columns:
            query[key] = value
    query_scaled = scaler.transform(query)
    query_weighted = query_scaled * weights_adjusted
    matrix_weighted = feature_matrix * weights_adjusted
    similarities = cosine_similarity(query_weighted, matrix_weighted)[0]
    occ = listings_rec['estimated_occupancy_l365d'].values
    occ_norm = (occ - occ.min()) / (occ.max() - occ.min())
    hybrid_scores = (alpha * similarities) + ((1 - alpha) * occ_norm)
    top_indices = hybrid_scores.argsort()[::-1][:top_n]
    results = listings_rec.iloc[top_indices][[
        'price', 'room_type',
        'neighbourhood_group_cleansed',
        'review_scores_rating',
        'accommodates',
        'minimum_nights',
        'estimated_occupancy_l365d'
    ]].copy()
    results['similarity_score'] = similarities[top_indices].round(3)
    results['hybrid_score'] = hybrid_scores[top_indices].round(3)
    return results.reset_index(drop=True)
# app layout
st.title('NYC Airbnb Listing Recommender')
st.write('Find listings that match your preferences.')

st.sidebar.header('Your Preferences')

price = st.sidebar.slider('Max price per night (USD)', 50, 1000, 150, step=25)
accommodates = st.sidebar.slider('Number of guests', 1, 16, 2)
room_type = st.sidebar.selectbox('Room type',
    ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'])
borough = st.sidebar.selectbox('Borough',
    ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'])
min_nights = st.sidebar.slider('Minimum nights', 1, 30, 2)
top_n = st.sidebar.slider('Number of recommendations', 3, 10, 5)

# build user query
user_prefs = {
    'price_log': np.log1p(price),
    'accommodates': accommodates,
    'review_scores_rating': 4.8,
    'has_reviews': 1,
    'minimum_nights': min_nights,
    f'room_type_{room_type}': 1,
    f'neighbourhood_group_cleansed_{borough}': 1
}

# get recommendations
results = recommend_hybrid(user_prefs, top_n=top_n)

# clean up display
results = results[['price', 'room_type', 'neighbourhood_group_cleansed',
                   'review_scores_rating', 'accommodates',
                   'minimum_nights', 'estimated_occupancy_l365d',
                   'similarity_score', 'hybrid_score']]

results['price'] = results['price'].astype(int)

results = results.rename(columns={
    'neighbourhood_group_cleansed': 'Borough',
    'room_type': 'Room Type',
    'review_scores_rating': 'Review Score',
    'estimated_occupancy_l365d': 'Occupancy (days)',
    'minimum_nights': 'Min Nights',
    'price': 'Price (USD)',
    'similarity_score': 'Similarity',
    'hybrid_score': 'Hybrid Score'
})

st.subheader(f'Top {top_n} recommendations')
st.dataframe(results)

st.write(f'Average occupancy: {results["Occupancy (days)"].mean():.1f} days')
st.write(f'Average review score: {results["Review Score"].mean():.3f}')