### Dataset
`ml_netflix.csv`: join on [ml-latest-small](https://grouplens.org/datasets/movielens/) and [Netflix data](https://www.kaggle.com/datasets/shivamb/netflix-shows?resource=download)
- code for generating the dataset in `utils/gen_data.ipynb`

### Experimented on
1. LDA
    - [pure Python](https://nicoleeesim97.medium.com/building-a-simple-content-based-recommender-system-for-movies-and-tv-shows-73fec4f325ae)
        - can run the notebook straight through
    - [Stan](https://mc-stan.org/docs/2_18/stan-users-guide/latent-dirichlet-allocation.html)
        - warning: takes ~1.67 hours to finish sampling on all 1056 movies
        - run `python3 stan-lda.py` in terminal with the following available flags:
            - `regen_words_df` (bool): True if you'd like to regenerate the dataframe mapping each word (ID) to a document/movie (ID)
                - saved to `cache/words_df.csv`
            - `regen_data_lemmatized` (bool): True if you'd like to regenerate the lemmatized movie descriptions
                - saved to `cached/data_lemmatized.txt`
            - `num_movies` (int): the first `num_movies` movies from the data set that you'd like to train on
                - by default, it's the number of movies in the data set (1056)
            - `just_eval` (bool): True if you'd like to just calculate the evaluation metrics. Assumes you already have the trained posterior values in `results/theta.npy`.
2. PMF
    - [pure Python](https://towardsdatascience.com/pmf-for-recommender-systems-cbaf20f102f0)
        - can run the notebook straight through
    - [Stan](https://discourse.mc-stan.org/t/bayesian-matrix-factorization/15142)
        - run `python3 stan-pmf.py` in terminal with the followiing available flags:
            - `just_eval` (bool): True if you'd like to just calculate the evaluation metrics. Assumes you already have the trained posterior values in `results/Z.npy` and `results/W.npy`.
### Eval Metrics
1. Personalization
2. MAP@K
3. Mean Precision
4. RMSE
