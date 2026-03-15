from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.ucb_algorithm import UCBBandit
from src.recommender import simulate_recommendations
from src.evaluation import calculate_metrics, plot_cumulative_reward, plot_regret, plot_selection_frequency, generate_html_report

def main():
    """
    Main function to run the full UCB Movie Recommendation pipeline.
    """
    # 1. Load dataset
    print("Loading data...")
    ratings, movies = load_data('dataset')
    
    # 2. Preprocess ratings
    print("Preprocessing data...")
    normalized_matrix = preprocess_data(ratings)
    
    n_movies = normalized_matrix.shape[1]
    n_users = normalized_matrix.shape[0]
    
    print(f"Dataset stats: {n_users} users, {n_movies} movies.")
    
    # 3. Initialize the UCB algorithm
    bandit = UCBBandit(n_movies)
    
    # 4. Run the UCB recommendation simulation
    print("Running UCB simulation (this may take a minute)...")
    num_steps = 100000 # Adjust this if you want a longer/shorter simulation
    
    # Simulating interactions
    rewards, optimal_rewards, selected_movies = simulate_recommendations(normalized_matrix, bandit, num_steps)
    
    # 5. Evaluate performance
    print("Calculating evaluation metrics...")
    cumulative_rewards, cumulative_regret = calculate_metrics(rewards, optimal_rewards)
    
    # Calculate some summary statistics
    total_reward = cumulative_rewards[-1]
    total_regret = cumulative_regret[-1]
    average_reward = total_reward / num_steps
    
    print(f"Finished simulation! \nTotal recommendations: {num_steps}\nTotal reward: {total_reward:.2f}\nAverage reward: {average_reward:.4f}")
    
    # Display top recommendations
    # Map index back to movie_id and title
    matrix_columns = normalized_matrix.columns
    top_indices = bandit.get_top_k_recommendations(k=10)
    print("\nTop 10 recommended movies:")
    
    top_movies_list = []
    
    for i in top_indices:
        movie_id = matrix_columns[i] # get back the original movie ID
        title_series = movies.loc[movies['movie_id'] == movie_id, 'title']
        title = title_series.values[0] if len(title_series) > 0 else 'Unknown'
        
        movie_data = {
            'id': movie_id,
            'title': title,
            'avg_reward': bandit.average_reward[i],
            'selections': bandit.n_selections[i]
        }
        top_movies_list.append(movie_data)
        
        print(f"- {title} (ID: {movie_id})")
        print(f"  Selections: {bandit.n_selections[i]}, Average Reward: {bandit.average_reward[i]:.4f}")
        
    # 6. Plot graphs
    print("\nPlotting results to 'results/' directory...")
    plot_cumulative_reward(cumulative_rewards)
    plot_regret(cumulative_regret)
    plot_selection_frequency(bandit)
    print("Plots generated successfully!")
    
    # 7. Generate HTML report
    print("\nGenerating HTML report...")
    generate_html_report(num_steps, total_reward, average_reward, top_movies_list)

if __name__ == "__main__":
    main()
