import numpy as np
import math

class UCBBandit:
    """
    Multi-Armed Bandit using the Upper Confidence Bound (UCB) algorithm.
    Each action represents recommending a specific movie.
    """
    
    def __init__(self, n_movies):
        """
        Initialize the bandit algorithm with n_movies arms (actions).
        """
        self.n_movies = n_movies
        self.t = 0 # Current timestep
        
        # Track number of times each movie is selected (N(a))
        self.n_selections = np.zeros(n_movies)
        
        # Track total rewards for each movie
        self.total_rewards = np.zeros(n_movies)
        
        # Average reward (Q(a))
        self.average_reward = np.zeros(n_movies)
        
    def select_movie(self):
        """
        Select the next movie to recommend based on the UCB value.
        Formula: UCB(a) = Q(a) + sqrt((2 * ln(t)) / N(a))
        """
        # We must explore all movies at least once to avoid division by zero
        unexplored_movies = np.where(self.n_selections == 0)[0]
        if len(unexplored_movies) > 0:
            # Randomly select among unexplored to break ties
            return np.random.choice(unexplored_movies)
            
        # Calculate UCB score for all movies since no unexplored ones exist
        # Q(a) [average_reward] + Exploration term
        exploration_term = np.sqrt((2 * math.log(self.t)) / self.n_selections)
        ucb_values = self.average_reward + exploration_term
        
        # Tie-breaking if multiple movies have the exact same UCB value
        max_value = np.max(ucb_values)
        best_movies = np.where(ucb_values == max_value)[0]
        
        return np.random.choice(best_movies)
        
    def update(self, movie_idx, reward):
        """
        Update the estimates for the selected movie after receiving a reward.
        """
        self.t += 1
        self.n_selections[movie_idx] += 1
        self.total_rewards[movie_idx] += reward
        self.average_reward[movie_idx] = self.total_rewards[movie_idx] / self.n_selections[movie_idx]
        
    def get_top_k_recommendations(self, k=5):
        """
        Utility method to view the current top recommended movies based on average reward.
        Returns the indices of the movies with highest average reward.
        """
        # Sort in descending order based on average reward (Q(a))
        top_indices = np.argsort(self.average_reward)[::-1]
        
        # Filter out movies that haven't been adequately explored (arbitrary threshold)
        # or just return the highest average rewards regardless.
        return top_indices[:k]
