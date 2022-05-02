### Scrape Scripts
[MovieLens](https://grouplens.org/datasets/movielens/)'s latest small dataset only had the movie names and genres. So to fuel the LDA model, we can scrape those movies' scripts from [this website](https://imsdb.com/). They're scraped based on movie titles in `movieTitle.txt`. Only about 200 non-null movies have been scraped. To get more, run the following in the project root:

1. To install package dependencies
    ```
    yarn
    ```

2.
    ```
    node getScripts.js
    ```

# The scraper is pretty jank so if you stop scraping midway, be sure to delete the already-saved movie titles from `movieTitle.txt`.

__Scripts are saved to `scripts.txt`.__

`movieTitle.txt` can be generated from scratch in `explore.ipynb`. Can also pull the non-null movies in `explore.ipynb`.