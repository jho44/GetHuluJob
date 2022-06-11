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
    - [Pyro](https://pyro.ai/examples/prodlda.html)
        - can run the notebook straight through
        - modify number of topics, number of epochs run, etc. in cell 4
    - [Turing](https://github.com/TuringLang/TuringExamples/blob/466fe443ea0270ff2a7420c30339b2bce7e8f4b9/nonparametrics/topic_model.jl)
        - can run the notebook using Julia runtime
        - results are output to CSV (`cache/julia_out.csv`) for evaluation in Python using `eval_julia.ipynb`
2. PMF
    - [pure Python](https://towardsdatascience.com/pmf-for-recommender-systems-cbaf20f102f0)
        - can run the notebook straight through
    - [Stan](https://discourse.mc-stan.org/t/bayesian-matrix-factorization/15142)
        - run `python3 stan-pmf.py` in terminal with the followiing available flags:
            - `just_eval` (bool): True if you'd like to just calculate the evaluation metrics. Assumes you already have the trained posterior values in `results/Z.npy` and `results/W.npy`.
### Eval Metrics
1. [Personalization](https://medium.com/qloo/popular-evaluation-metrics-in-recommender-systems-explained-324ff2fb427d)
2. [MAP@K](https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html)
3. [Mean Precision](https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52)
4. RMSE
