#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import ast # Used to convert string representation of list to actual list
import streamlit as st
import pickle



# In[7]:


# Load the datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge the two dataframes on the 'title' column
movies = movies.merge(credits, on='title')

# Select the relevant columns for the recommendation model
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# --- Data Preprocessing ---

# Remove rows with missing data
movies.dropna(inplace=True)

# Helper function to extract names from JSON-like strings
def convert(text):
    """Converts a JSON-like string of a list of dictionaries to a list of names."""
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# Apply the conversion to 'genres' and 'keywords' columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Helper function to get the top 3 actors' names
def convert_cast(text):
    """Extracts the names of the first 3 cast members."""
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# Helper function to fetch the director's name from the crew
def fetch_director(text):
    """Extracts the director's name from the crew information."""
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break # We only need one director
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Convert the 'overview' string into a list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# --- Feature Engineering ---

# Function to remove spaces between words (e.g., "Science Fiction" -> "ScienceFiction")
# This helps the model treat it as a single entity
def remove_space(word_list):
    L = []
    for i in word_list:
        L.append(i.replace(" ", ""))
    return L

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

# Combine all the processed features into a single "tags" column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new dataframe with the essential columns
new_df = movies[['movie_id', 'title', 'tags']]

# Convert the list of tags into a single string for vectorization
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
# Convert tags to lowercase for consistency
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# --- Vectorization and Similarity Calculation ---

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the CountVectorizer. max_features limits the vocabulary size.
# stop_words removes common English words (like 'the', 'a', 'in')
cv = CountVectorizer(max_features=5000, stop_words='english')

# Transform the 'tags' column into a matrix of word counts
vectors = cv.fit_transform(new_df['tags']).toarray()

# Calculate the cosine similarity between all movie vectors
similarity = cosine_similarity(vectors)


# --- Saving the Model ---

import pickle

# Save the processed dataframe and the similarity matrix to files
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Model and data saved successfully!")


# In[8]:


# Function to recommend movies
def recommend(movie):
    """Finds the top 5 movies most similar to the selected one."""
    try:
        # Get the index of the selected movie
        movie_index = movies[movies['title'] == movie].index[0]
        # Get the similarity scores for that movie
        distances = similarity[movie_index]
        # Sort the movies based on similarity and get the top 5
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        for i in movies_list:
            # Get the title of the recommended movie by its index
            recommended_movies.append(movies.iloc[i[0]].title)
        return recommended_movies
    except IndexError:
        return ["Movie not found. Please select another one."]
    except Exception as e:
        return [f"An error occurred: {e}"]


# --- Load the pre-processed data ---

# Load the movie dictionary and similarity matrix from the files created by model.py
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))


# --- Streamlit Web App Interface ---

st.title('🎬 Movie Recommender System')

# Create a dropdown select box with all the movie titles
selected_movie_name = st.selectbox(
    'Select a movie you like, and we will recommend similar ones:',
    movies['title'].values)

# Add a button to trigger the recommendation
if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    st.write("Here are your recommendations:")
    
    # Display the list of recommended movies
    for movie_title in recommendations:
        st.success(movie_title)


# In[9]:


streamlit hello


# In[ ]:





# In[ ]:




