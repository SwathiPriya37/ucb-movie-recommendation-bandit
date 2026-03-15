import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_metrics(rewards, optimal_rewards):
    """
    Computes cumulative reward and regret over time.
    """
    # Cumulative sum of obtained rewards
    cumulative_rewards = np.cumsum(rewards)
    
    # Cumulative sum of optimal rewards
    cumulative_optimal_rewards = np.cumsum(optimal_rewards)
    
    # Regret = Optimal Reward - Obtained Reward at each step
    regret = optimal_rewards - rewards
    cumulative_regret = np.cumsum(regret)
    
    return cumulative_rewards, cumulative_regret

def plot_cumulative_reward(cumulative_rewards, save_path="results/cumulative_reward.png"):
    """
    Plots and saves the Cumulative Reward vs Time graph.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_rewards, label='UCB Cumulative Reward', color='blue')
    plt.xlabel('Timesteps')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward vs Time')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_regret(cumulative_regret, save_path="results/regret_curve.png"):
    """
    Plots and saves the Regret vs Time graph.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_regret, label='UCB Cumulative Regret', color='red')
    plt.xlabel('Timesteps')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret vs Time')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_selection_frequency(bandit, save_path="results/movie_frequency.png"):
    """
    Plots and saves the Movie Selection Frequency histogram.
    """
    plt.figure(figsize=(12, 6))
    # Count the number of selections for each movie (action)
    counts = bandit.n_selections
    
    # Plot only the top 50 selected movies to keep the plot readable
    top_indices = np.argsort(counts)[-50:]
    top_counts = counts[top_indices]
    
    plt.bar(range(len(top_counts)), top_counts, color='green')
    plt.xlabel('Movie Arms (Top 50 Most Selected)')
    plt.ylabel('Selection Frequency')
    plt.title('Movie Selection Frequency')
    plt.xticks(range(len(top_counts)), top_indices, rotation=90)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def generate_html_report(total_recs, total_reward, avg_reward, top_movies_list, save_path="results/report.html"):
    """
    Generates an HTML report summarizing the findings.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    movies_html = "<ul>\n"
    for movie in top_movies_list:
        movies_html += f"        <li>{movie['title']} (ID: {movie['id']}) - Selected {movie['selections']} times, Avg Reward: {movie['avg_reward']:.4f}</li>\n"
    movies_html += "    </ul>"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Recommendation System Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f4f4f4; }}
        .container {{ width: 80%; margin: auto; overflow: hidden; background: white; padding: 20px; box-shadow: 0 0 10px #ccc; }}
        h1, h2, h3 {{ color: #333; border-bottom: 2px solid #ccc; padding-bottom: 5px; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin-bottom: 20px; }}
        .metric-box {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Project Title: Movie Recommendation System using UCB Optimization</h1>

        <h2>Section 1: Project Overview</h2>
        <p>This project uses the Multi-Armed Bandit framework with the Upper Confidence Bound (UCB) strategy to recommend movies. The algorithm balances exploring less-known movies with exploiting highly-rated favorites.</p>

        <h2>Section 2: Dataset Explanation</h2>
        <p>The system leverages the **MovieLens 100K dataset** by GroupLens Research. It contains 100,000 ratings from 943 users across 1,682 movies. During preprocessing, ratings &ge; 4 were assigned a reward of 1, and ratings &lt; 4 were assigned a reward of 0.</p>

        <h2>Section 3: Algorithm Explanation</h2>
        <p>The UCB formula tracks average reward and explores actions sequentially:</p>
        <p><strong>UCB(a) = Q(a) + &radic;((2 * ln(t)) / N(a))</strong></p>
        <p>Where <em>Q(a)</em> is the average reward, <em>t</em> is the timestep, and <em>N(a)</em> is the total number of recommendations for movie <em>a</em>.</p>

        <h2>Section 4: Results and Graphs</h2>
        <div class="metric-box">
            <p><strong>Total Recommendations Simulated:</strong> {total_recs}</p>
            <p><strong>Total Cumulative Reward:</strong> {total_reward:.2f}</p>
            <p><strong>Average Target Reward Rate:</strong> {avg_reward:.4f}</p>
        </div>
        
        <h3>Cumulative Reward vs. Time</h3>
        <img src="cumulative_reward.png" alt="Cumulative Reward Graph" />

        <h3>Regret Curve vs. Time</h3>
        <img src="regret_curve.png" alt="Regret Curve Graph" />

        <h3>Movie Recommendation Frequency</h3>
        <img src="movie_frequency.png" alt="Movie Selection Frequency Graph" />

        <h2>Section 5: Top Recommended Movies</h2>
        {movies_html}
    </div>
</body>
</html>
"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML Report generated at {save_path}")
