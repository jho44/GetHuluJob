from __future__ import division
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('ggplot')

np.random.seed(1234)
import stan
import argparse
from sklearn.metrics import mean_squared_error
from utils.utils import get_users_faves
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--just_eval', default=False, type=bool)

args = parser.parse_args()

df = pd.read_csv('./ml_netflix.csv')

if not args.just_eval:
  # this part is Danning's child
  program = """
  data {
      int<lower=0> n_users;
      int<lower=0> n_movies;
      int<lower=0> n_features;
      int<lower=1> n_entries;
      array[n_entries] int<lower=0, upper=n_users> ii;
      array[n_entries] int<lower=0, upper=n_movies> jj;
      vector[n_entries] rating;
  }

  parameters {
      matrix<lower=0>[n_movies, n_features] W;
      matrix<lower=0>[n_users, n_features] Z;
      real<lower=0> beta;
  }

  model {
      for(n in 1:n_entries) {
          rating[n] ~ normal(W[jj[n], :] * Z[ii[n], :]', 1/sqrt(beta));
      }

      beta ~ gamma(1.5, 1);
      for(n in 1:n_users){
          Z[n,:] ~ exponential(1);
      }
      for(n in 1:n_movies){
          W[n,:] ~ exponential(1);
      }
  }
  """

  df['movieId'] += 1
  df['userId'] += 1
  n_users = len(df['userId'].unique())
  n_movies = len(df['movieId'].unique())
  n_entries = df.shape[0]
  n_features = 10
  rating = list(df['rating'])

  data = {
      'n_users': n_users,
      'n_movies': n_movies,
      'n_features': n_features,
      'n_entries': n_entries,
      'rating': rating,
      'ii': list(df['userId']),
      'jj': list(df['movieId'])
  }

  posterior = stan.build(program, data=data, random_seed=1)
  fit = posterior.sample(num_chains=1, num_samples=5)
  if not os.path.isdir("results"):
    os.mkdir("results")
  with open('results/Z.npy', 'wb') as f:
    np.save(f, fit['Z'])
  with open('results/W.npy', 'wb') as f:
    np.save(f, fit['W'])
  with open('results/beta.npy', 'wb') as f:
    np.save(f, fit['beta'])
else:
  # read posterior vals from file
  Z = np.load('results/Z.npy')
  W = np.load('results/W.npy')

  # average over samples
  Z = np.sum(Z, axis=2) / Z.shape[2]
  W = np.sum(W, axis=2) / W.shape[2]

  num_users = Z.shape[0]
  num_movies = W.shape[0]

  R = Z @ W.T
  # get min and max ratings to standardize predictions on 0.0 to 5.0 scale
  min_rating = np.min(R)
  max_rating = np.max(R)

  def predict(user_id, movie_id):
    r_ij = R[user_id][movie_id]
    return 0 if max_rating == min_rating else ((r_ij - min_rating) / (max_rating - min_rating)) * 5.0

  def rmse_evaluate(dataset):
    ground_truths = []
    predictions = []

    for index, row in dataset.iterrows():
        ground_truths.append(row.loc['rating'])
        predictions.append(predict(row.loc['userId'], row.loc['movieId']))

    return mean_squared_error(ground_truths, predictions, squared=False)

  def get_recs(dataset):
    all_predictions = {}

    user_5_df, user_4_df = get_users_faves(dataset)

    for user in range(num_users):
        num_actual_likes = len(user_5_df[user_5_df['userId']==user])
        # if no 5 ratings, find number of 4 ratings
        if num_actual_likes == 0:
            num_actual_likes = len(user_4_df[user_4_df['userId']==user])
        predictions = np.zeros((num_movies, 1))

        for movie in range(num_movies):
            predictions[movie] = predict(user, movie)

        indices = np.argsort(-predictions, axis=0)
        predicted_likes = []
        for j in range(num_actual_likes):
            movie_id = indices[j][0]
            predicted_likes.append(movie_id)
        all_predictions[user] = predicted_likes

    return all_predictions

  def mapk(recommendations, user_5_df, user_4_df):
    '''
    slightly different from the one in utils.utils because this workflow has train-test datasets
    and not every user is in both datasets
    '''
    # calculate MAP @ k
    # http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html

    # for each user
    map_k = 0
    for user in recommendations:
        l = len(recommendations[user])
        user_actual = user_5_df[user_5_df['userId']==user]['movieId']
        if user_actual.empty:
            user_actual = user_4_df[user_4_df['userId']==user]['movieId']
        if user_actual.empty:
            continue
        sum = 0
        if l == 0:
            continue
        for k in range(1, l):
            # top k recommendations and top k actually rated movies for user
            user_rec = recommendations[user][:k]
            user_actual_k = set(user_actual[:k])
            # only add if the kth item was relevant
            if user_rec[-1] in user_actual_k:
                # find intersections
                user_rec = set(user_rec)
                intersection = list(user_rec & user_actual_k)
                # add precision to sum
                sum += len(intersection) / float(k)
        # divide by min(m, N)
        sum /= float(l)
        map_k += sum

    # print final map@k value
    map_k /= float(len(recommendations))
    print(f'MAP@K: {map_k}')
    return map_k

  def calc_personalization(recommendations, num_movies, num_users): # nikki's
    # calculate personalization matrix
    # https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093
    # create num_user x num_movies
    personalization = np.zeros((num_users, num_movies))

    for row in recommendations:
        user = row
        for movie_ind in recommendations[row]:
            personalization[user][movie_ind] = 1

    cosine_sim = cosine_similarity(personalization, personalization)

    # compute average of upper triangle to get cosine similarity
    sum = 0
    denom = (num_users-1)*(num_users)/2
    for i in range(num_users-1):
        for j in range(i+1, num_users):
            sum += cosine_sim[i][j]

    similarity = sum/denom
    dissimilarity = 1 - similarity
    print(f'Personalization: {dissimilarity}')
    return dissimilarity

  def mean_precision(recommendations, num_users, user_5_df, user_4_df): # mostly nikki's
    # http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html

    # for each user
    metric = 0
    for user in range(num_users):
        l = len(recommendations[user])
        user_actual = user_5_df[user_5_df['userId']==user]['movieId']
        if user_actual.empty:
            user_actual = user_4_df[user_4_df['userId']==user]['movieId']
        if user_actual.empty:
            continue

        if l == 0:
            continue

        user_actual = set(user_actual.values)
        # top k recommendations and top k actually rated movies for user
        user_rec = set(recommendations[user])
        # find intersections
        intersection = list(user_rec & user_actual)
        # add precision to sum
        metric += len(intersection) / float(l)

    metric /= float(num_users)
    print(f'MEAN PRECISION (NOT mAP): {metric}')
    return metric

  def eval(recs, user_5_df, user_4_df, metric_type):
    if metric_type == 'MAPK':
      return mapk(recs, user_5_df, user_4_df)
    elif metric_type == 'PER':
      return calc_personalization(recs, num_movies, num_users)
    elif metric_type == 'MEAN_PRECISION':
      return mean_precision(recs, num_users, user_5_df, user_4_df)

  print('RMSE of training set:', rmse_evaluate(df))

  recs = get_recs(df)
  user_5_df, user_4_df = get_users_faves(df)
  eval(recs, user_5_df, user_4_df, 'MAPK')
  eval(recs, user_5_df, user_4_df, 'PER')
  eval(recs, user_5_df, user_4_df, 'MEAN_PRECISION')