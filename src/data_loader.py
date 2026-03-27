import pandas as pd
import os

def load_data(data_dir='dataset'):
    """
    Load the MovieLens dataset using pandas.
    Parses u.data with tab separator and assigns column names.
    """
    data_path = os.path.join(data_dir, 'u.data')
    item_path = os.path.join(data_dir, 'u.item')
   
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    # Load ratings
    # u.data format: user_id | movie_id | rating | timestamp
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(data_path, sep='\t', names=ratings_cols, encoding='latin-1')
    
    # Load movies (items)
    # The u.item file has many columns, but we mainly care about movie_id and title
    item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
                 'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
                 
    movies_df = pd.read_csv(item_path, sep='|', names=item_cols, encoding='latin-1', usecols=['movie_id', 'title'])
    
    return ratings_df, movies_df

if __name__ == "__main__":
    # Test the loader
    ratings, movies = load_data()
    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    print("\nFirst 5 ratings:")
    print(ratings.head())
    print("\nFirst 5 movies:")
    print(movies.head())
