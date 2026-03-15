# Movie Recommendation System using Upper Confidence Bound (UCB) Optimization

## Project Overview
This project implements a Movie Recommendation System using a Multi-Armed Bandit algorithm with the Upper Confidence Bound (UCB) strategy for a Numerical Optimization course. The goal is to recommend movies to users while balancing exploration (trying out less-recommended movies to determine their true value) and exploitation (recommending movies known to yield high user ratings).

The system simulates the recommendation sequence over many timesteps, using historical rating data to emulate user responses.

## Dataset Description
This project uses the **MovieLens 100K dataset** by GroupLens Research. The dataset contains 100,000 ratings (1-5) from 943 users on 1,682 movies.

- `dataset/u.data`: Contains the rating information in the format `user_id | movie_id | rating | timestamp`.
- `dataset/u.item`: Contains movie metadata, including titles and varying genres.

*The raw ratings are normalized to a scale of `0.0 - 1.0` during preprocessing to align with standard Bandit reward ranges.*

## UCB Algorithm Explanation
In the Multi-Armed Bandit problem, we treat each movie recommendation as an "arm" to pull. The Upper Confidence Bound (UCB) algorithm ensures we don't just greedily pick the best-known movie, but also explore movies we are uncertain about. 

The UCB formula for action/movie $a$ is:

$$ UCB(a) = Q(a) + \sqrt{\frac{2 \ln(t)}{N(a)}} $$

Where:
- $Q(a)$: The average observed reward (rating) for action $a$.
- $t$: The current timestep in the simulation.
- $N(a)$: The number of times action $a$ has been selected.

This strategy ensures that the exploration term ($\sqrt{\frac{2 \ln(t)}{N(a)}}$) grows when a movie hasn't been picked often, increasing its chance of being selected and effectively balancing exploration with exploitation.

## Project Structure
```
ucb-movie-recommendation/
│
├── dataset/                  # Contains u.data and u.item (MovieLens 100K)
│
├── src/
│   ├── data_loader.py      # Loads the dataset and parses it into pandas DataFrames
│   ├── preprocessing.py    # Generates and normalizes the user-movie rating matrix
│   ├── ucb_algorithm.py    # Implements the UCBBandit logic and mathematics
│   ├── recommender.py      # Simulates the timestep-based recommendation process
│   └── evaluation.py       # Metrics and graph generation routines
│
├── results/                  # Generated plots are saved here
│
├── main.py                   # Main execution pipeline
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## How to Run the Project

1. **Install Dependencies:**
   Ensure you have Python 3 installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Simulation:**
   Ensure `u.data` and `u.item` are present in the `dataset/` directory. Then simply execute:
   ```bash
   python main.py
   ```

3. **View Results:**
   The output will display the overall metrics (Cumulative Reward, Cumulative Regret) and the top 5 recommended movies list.
   Check the `results/` folder for generated visual evaluations:
   - `cumulative_reward.png`
   - `regret.png`
   - `selection_frequency.png`
