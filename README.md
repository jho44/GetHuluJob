### Dataset
`ml_netflix.csv`: join on [ml-latest-small](https://grouplens.org/datasets/movielens/) and [Netflix data](https://www.kaggle.com/datasets/shivamb/netflix-shows?resource=download)
- code for generating the dataset in `utils/gen_data.ipynb`

### Experimented on
1. LDA
    - [pure Python](https://nicoleeesim97.medium.com/building-a-simple-content-based-recommender-system-for-movies-and-tv-shows-73fec4f325ae)
    - [STAN](https://mc-stan.org/docs/2_18/stan-users-guide/latent-dirichlet-allocation.html)
        - Haven't gotten the STAN model to handle very many movies -- sampling is slow past 50 movies.

### Eval Metrics
1. Personalization
2. MAP@K
