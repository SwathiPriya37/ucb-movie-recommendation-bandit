import numpy as np

def simulate_recommendations(rating_matrix, bandit, num_steps=10000):
    """
    Simulates the recommendation process over a series of timesteps.
    
    Parameters:
    - rating_matrix: Normalized user-movie rating matrix (numpy array or pandas DataFrame)
    - bandit: The initialized UCBBandit instance
    - num_steps: Number of simulation steps (timesteps)
    
    Returns:
    - rewards: Array containing the reward received at each timestep
    - optimal_rewards: Array containing the maximum possible reward at each timestep
    - selected_movies: Array containing the index of the selected movie at each timestep
    """
    # Convert rating matrix to numpy array for faster indexing if it's a DataFrame
    if hasattr(rating_matrix, 'values'):
        matrix_values = rating_matrix.values
    else:
        matrix_values = rating_matrix
        
    num_users = matrix_values.shape[0]
    
    rewards = np.zeros(num_steps)
    optimal_rewards = np.zeros(num_steps)
    selected_movies = np.zeros(num_steps, dtype=int)
    
    for t in range(num_steps):
        # 1. Randomly pick a user for this timestep
        user_idx = np.random.randint(0, num_users)
        
        # 2. Bandit selects a movie
        # To strictly follow standard MAB where actions are independent of users (for simplicity),
        # we ask the bandit which movie to recommend.
        movie_idx = bandit.select_movie()
        
        # 3. Get the actual reward from the user-movie matrix
        reward = matrix_values[user_idx, movie_idx]
        
        # 4. Find the optimal reward (what the best movie for this user would have given)
        optimal_reward = np.max(matrix_values[user_idx, :])
        
        # 5. Update the bandit with the received reward
        bandit.update(movie_idx, reward)
        
        # Track metrics
        rewards[t] = reward
        optimal_rewards[t] = optimal_reward
        selected_movies[t] = movie_idx
        
        # Optional: Print progress
        if (t + 1) % 5000 == 0:
            print(f"Step {t + 1}/{num_steps} completed.")
            
    return rewards, optimal_rewards, selected_movies
