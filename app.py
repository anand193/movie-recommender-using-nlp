import streamlit as st
import pickle
import pandas as pd
import os
import urllib.request

# ----------------------------
# Load movie_dict.pkl
# ----------------------------
MOVIE_URL = "https://huggingface.co/anandmehto/Movie_recommendation_system/resolve/main/movie_dict.pkl?download=true"
if not os.path.exists("movie_dict.pkl"):
    st.write("Downloading movie_dict.pkl from Hugging Face...")
    urllib.request.urlretrieve(MOVIE_URL, "movie_dict.pkl")

movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

# ----------------------------
# Load similarity.pkl
# ----------------------------
SIMILARITY_URL = "https://huggingface.co/anandmehto/Movie_recommendation_system/resolve/main/similarity.pkl?download=true"
if not os.path.exists("similarity.pkl"):
    st.write("Downloading similarity.pkl from Hugging Face...")
    urllib.request.urlretrieve(SIMILARITY_URL, "similarity.pkl")

similarity = pickle.load(open("similarity.pkl", "rb"))

# ----------------------------
# Streamlit App
# ----------------------------
st.title('🎬 Movie Recommendation System')

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    recommended_movies = [movies.iloc[i[0]].title for i in movies_list]
    return recommended_movies

selected_movie_name = st.selectbox('Select a Movie', movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    st.subheader("Recommended Movies:")
    for i in recommendations:
        st.write(i)