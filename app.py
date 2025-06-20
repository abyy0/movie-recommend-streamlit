import streamlit as st
import pandas as pd
import difflib
import base64
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie

# ✅ Function to set background image
def set_bg(image_file_path):
    with open(image_file_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

# ✅ Function to load Lottie animation
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ✅ Set background
set_bg("minion.jpeg")  # Make sure this is uploaded

# ✅ Style title
st.markdown("""
<style>
h1 {
    color: white !important;
    animation: fadeIn 2s ease-in;
    text-shadow: 2px 2px 4px black;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# ✅ Show animated movie banner
lottie_movie = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_LkK1Xk.json")
st_lottie(lottie_movie, height=200)

# ✅ Title
st.title("🎬 Movie Recommendation System")

# ✅ Load movie data
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

movies_data = load_data()

# ✅ Feature extraction
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data[selected_features].agg(' '.join, axis=1)
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# ✅ Input
movie_name = st.text_input("Enter a movie name you like:")

# ✅ Recommend button
if st.button("Recommend"):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)[1:6]

        st.success(f"Movies similar to '{close_match}':")
        for i, movie in enumerate(sorted_similar_movies):
            st.write(f"{i+1}. {movies_data.iloc[movie[0]].title}")
    else:
        st.error("No matching movie found. Try another one.")
