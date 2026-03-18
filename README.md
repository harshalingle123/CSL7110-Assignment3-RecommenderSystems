# CSL7110 Assignment 3: Recommender Systems

## Content-Based and Collaborative Filtering

This assignment implements various recommender system techniques on the MovieLens dataset.

## Dataset

- MovieLens Small (ml-latest-small) - Already included in repository
- 100,836 ratings from 610 users on 9,742 movies
- Files: `movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`

## Requirements

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
pip install scikit-surprise  # For Task 6
pip install tensorflow       # For Task 8
pip install lime             # For Task 12 (optional)
```

## Assignment Structure (120 Marks Total)

The notebook is organized into 6 parts with 13 tasks:

### Part 1: Content-Based Filtering 
- Task 1: TF-IDF Based Recommendation
- Task 2: User-Profile-Based Content Recommender

### Part 2: Collaborative Filtering 
- Task 3: User-Based Collaborative Filtering
- Task 4: Item-Based Collaborative Filtering

### Part 3: Matrix Factorization 
- Task 5: Implementing SVD for Recommendations
- Task 6: Matrix Factorization with Surprise Library

### Part 4: Hybrid Recommendation Model 
- Task 7: Meta-learning Hybrid Model

### Part 5: Learning-Based Recommender Systems 
- Task 8: Content-Based Filtering with Neural Networks
- Task 9: Reinforcement Learning in Recommender Systems

### Part 6: Explainability 
- Task 10: Feature-Based Explanations (SHAP)
- Task 11: Neighborhood-Based Explanations
- Task 12: Model-Agnostic Explainability (LIME)
- Task 13: Evaluating Explainability

## How to Run

1. Ensure the dataset is in the `ml-latest-small/` folder (already included)
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook CSL7110_Assignment3_RecommenderSystems.ipynb
   ```
3. Run all cells sequentially (Kernel > Restart & Run All)

## Evaluation Metrics

- RMSE (Root Mean Squared Error) - For rating prediction accuracy
- Precision@K - Fraction of recommended items that are relevant
- Recall@K - Fraction of relevant items that are recommended

## Key Assumptions

1. Rating Threshold: Movies rated >= 4.0 are considered "liked"
2. Cold-Start Users: Defined as having < 10 ratings
3. Similarity Metrics:
   - Cosine similarity for TF-IDF vectors
   - Pearson correlation considered for handling user rating biases
4. RL Rewards:
   - Positive (+1) for ratings >= 4
   - Negative (-1) for ratings < 4
   - Neutral (0) for unrated movies
5. Missing Values: Filled with user mean for SVD factorization

## Notes

- Some cells may take several minutes to run (especially hyperparameter tuning in Task 6)
- TensorFlow warnings can be safely ignored
- If LIME is not installed, Task 12 will display a message and skip gracefully
- The notebook saves no external files - all outputs are displayed inline

## Methods Implemented

| Method | Type | Key Technique |
|--------|------|---------------|
| TF-IDF CBF | Content-Based | Genre vectorization |
| User-Profile CBF | Content-Based | Weighted genre preferences |
| User-Based CF | Collaborative | User similarity |
| Item-Based CF | Collaborative | Item similarity |
| SVD | Matrix Factorization | Latent factors |
| Surprise SVD | Matrix Factorization | Optimized SGD |
| Hybrid | Meta-learning | ML combination |
| Neural Network | Deep Learning | Learned embeddings |
| MAB/Q-Learning | Reinforcement Learning | Online learning |
