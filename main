import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google_genai as genai


# Set up Google GenAI Client for Gemini 2.0
client = genai.Client(api_key="AIzaSyCGWgWOm8Z5Btgm6Jat4C0MQKeTQm-I3C4")

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define course catalog
course_catalog = [
    {"name": "Data Science", "description": "Learn data analysis, machine learning, and statistics."},
    {"name": "Product Management", "description": "Master product strategy, roadmaps, and team leadership."},
    {"name": "Python Programming", "description": "Start coding with Python for various applications."},
    {"name": "Marketing Analytics", "description": "Use data-driven strategies to optimize marketing campaigns."},
    {"name": "UX/UI Design", "description": "Design intuitive user experiences and interfaces."}
]

# Precompute course embeddings
course_embeddings = embedding_model.encode([course["description"] for course in course_catalog])

def recommend_course(user_query):
    """Recommend a course based on user query."""
    # Embed the user query
    query_embedding = embedding_model.encode(user_query)

    # Compute similarities
    similarities = cosine_similarity([query_embedding], course_embeddings)[0]

    # Find the best match
    best_match_idx = np.argmax(similarities)
    best_match = course_catalog[best_match_idx]
    return best_match

def chat_with_student_gemini(user_input):
    """Generate a conversational response using Gemini 2.0."""
    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=user_input
    )
    return response.text

# Streamlit interface
st.title("Course Recommender")
st.write("Find the best course for your needs!")

user_query = st.text_input("What are you looking to learn?")

if user_query:
    # Recommend a course
    recommended_course = recommend_course(user_query)

    # Generate a conversational response using Gemini 2.0
    gemini_response = chat_with_student_gemini(user_query)

    # Display the results
    st.subheader("Recommended Course")
    st.write(f"**{recommended_course['name']}**: {recommended_course['description']}")

    st.subheader("AI Assistant Response")
    st.write(gemini_response)
