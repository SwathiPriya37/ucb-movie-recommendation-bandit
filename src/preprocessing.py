import pandas as pd
import numpy as np

def create_rating_matrix(ratings_df):
    """
    Creates a user-movie rating matrix from the ratings dataframe.
    Rows represent users, columns represent movies, and values are the ratings.
    """
    rating_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating')
    
    # Fill missing ratings with 0 for the sake of the bandit simulation
    # In a real scenario, an unrated movie would yield 0 reward if recommended
    rating_matrix = rating_matrix.fillna(0)
    
    return rating_matrix

def normalize_ratings(rating_matrix):
    """
    Converts the ratings into a binary reward matrix:
    Ratings >= 4 are treated as reward = 1.
    Ratings < 4 (and unrated 0s) are treated as reward = 0.
    """
    binary_matrix = (rating_matrix >= 4.0).astype(float)
    return binary_matrix

def preprocess_data(ratings_df):

    """
    Full preprocessing pipeline.
    """
    print("Creating rating matrix...")
    rating_matrix = create_rating_matrix(ratings_df)
    
    print("Normalizing ratings...")
    normalized_matrix = normalize_ratings(rating_matrix)
    
    return normalized_matrix

if __name__ == "__main__":
    # Simple test assuming data_loader is available
    from data_loader import load_data
    ratings, _ = load_data()
    norm_matrix = preprocess_data(ratings)
    print(f"Matrix shape: {norm_matrix.shape}")
    print("Preview of normalized matrix:")
    print(norm_matrix.iloc[:5, :5])
