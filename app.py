import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================
# LOAD DATA
# ==========================================

df = joblib.load("movies_data.pkl")
similarity_matrix = joblib.load("similarity_matrix.pkl")


# ==========================================
# USER HISTORY
# ==========================================

def get_user_history(user_id):

    history = df[df["User_ID"] == user_id]

    return history


# ==========================================
# USER RECOMMENDATIONS
# ==========================================

def recommend_for_user(user_id, num_recommendations=5):

    # Get user's previous movies
    user_history = df[df["User_ID"] == user_id]

    if user_history.empty:
        return []

    # Get indices of watched movies
    watched_indices = user_history.index.tolist()

    # Calculate similarity to user's history
    recommendation_scores = similarity_matrix[
        watched_indices
    ].mean(axis=0)

    # Don't recommend movies already watched
    recommendation_scores[
        watched_indices
    ] = -1

    # Sort recommendations
    recommended_indices = np.argsort(
        recommendation_scores
    )[::-1]

    # Get top movies
    recommended_indices = recommended_indices[
        :num_recommendations
    ]

    recommendations = []

    for index in recommended_indices:

        movie_name = df.iloc[index]["Movie_Title"]

        score = recommendation_scores[index]

        recommendations.append(
            (movie_name, score)
        )

    return recommendations


# ==========================================
# STREAMLIT PAGE
# ==========================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)


st.title("🎬 Movie Recommendation System")

st.write(
    "Get movie recommendations based on your "
    "previously watched movies."
)


# ==========================================
# SELECT USER
# ==========================================

user_ids = sorted(
    df["User_ID"].dropna().unique()
)

selected_user = st.selectbox(
    "Select User ID:",
    user_ids
)


# ==========================================
# SHOW WATCH HISTORY
# ==========================================

user_history = get_user_history(
    selected_user
)

st.subheader("📺 Previously Watched")

if not user_history.empty:

    for _, row in user_history.iterrows():

        st.write(
            f"🎬 **{row['Movie_Title']}** "
            f"| Genre: {row['Genre']} "
            f"| Rating: {row['Rating']}"
        )

else:

    st.info("No viewing history found.")


# ==========================================
# NUMBER OF RECOMMENDATIONS
# ==========================================

number = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=10,
    value=5
)


# ==========================================
# RECOMMEND BUTTON
# ==========================================

if st.button("🎥 Recommend Movies"):

    recommendations = recommend_for_user(
        selected_user,
        number
    )

    st.subheader(
        "🍿 Recommended Movies"
    )

    if recommendations:

        for rank, (movie, score) in enumerate(
            recommendations,
            start=1
        ):

            st.write(
                f"**{rank}. {movie}** "
                f"— Similarity: {score:.3f}"
            )

    else:

        st.warning(
            "Not enough viewing history for this user."
        )