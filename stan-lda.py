import stan
import pandas as pd
import numpy as np

import re
import os

# LDA Model
import gensim
from gensim.utils import simple_preprocess
import spacy

# NLTK Stop words
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

import argparse
import ast
from utils.utils import get_recs_for_all_users, get_users_faves, sort_probs, get_helper_vals, calc_personalization, mapk

"""
GOAL: recommender system for movies
"""

program = """
data {
  int<lower=2> K;               // num topics
  int<lower=2> V;               // num words
  int<lower=1> M;               // num docs
  int<lower=1> N;               // total word instances
  array[N] int<lower=1,upper=V> w;    // word n
  array[N] int<lower=1,upper=M> doc;  // doc ID for word n
  vector<lower=0>[K] alpha;     // topic prior
  vector<lower=0>[V] beta;      // word prior
}
parameters {
  array[M] simplex[K] theta;   // topic dist for doc m
  array[K] simplex[V] phi;     // word dist for topic k
}
model {
  for (m in 1:M)
    theta[m] ~ dirichlet(alpha);  // prior
  for (k in 1:K)
    phi[k] ~ dirichlet(beta);     // prior
  for (n in 1:N) {
    array[K] real gamma;
    for (k in 1:K)
      gamma[k] = log(theta[doc[n], k]) + log(phi[k, w[n]]);
    target += log_sum_exp(gamma);  // likelihood;
  }
}
"""
parser = argparse.ArgumentParser()
parser.add_argument('--regen_words_df', default=False, type=bool)
parser.add_argument('--regen_data_lemmatized', default=False, type=bool)
parser.add_argument('--num_movies', default=888, type=int)
parser.add_argument('--just_eval', default=False, type=bool)

args = parser.parse_args()

if args.just_eval and (args.regen_words_df or args.regen_data_lemmatized):
  raise Exception("just_eval being true means you don't want to train, but you indicated that you'd like to regen the datasets for training")

NUM_MOVIES = args.num_movies
NUM_TOPICS = 25

def eval(movie_probs):
  thing = np.sum(movie_probs, axis=2) / movie_probs.shape[2]

  movie_probs = np.max(thing, axis=1)
  movie_topics = np.argmax(thing, axis=1)

  data = {'Topic': movie_topics, 'Probability': movie_probs}
  movie_topics_probs = pd.DataFrame(data)

  movies_ratings = pd.read_csv('./ml_netflix.csv')
  unique_movies = movies_ratings.drop_duplicates('title')[['title', 'movieId']].reset_index(drop=True)

  movie_topics_probs = movie_topics_probs.merge(unique_movies, left_index=True, right_index=True)
  movies_ratings_probs = movies_ratings.merge(movie_topics_probs, how='left', on=['title', 'movieId'])
  movies_ratings_probs.dropna(inplace=True)

  movies_ratings_probs = movies_ratings_probs.astype({'Topic': 'int32'})

  # find all 4 and 5 star ratings
  user_5_df, user_4_df = get_users_faves(movies_ratings_probs)

  per_topic = sort_probs(int(NUM_TOPICS), movie_topics_probs)

  num_movies, num_users = get_helper_vals(movies_ratings_probs)
  recommendations = get_recs_for_all_users(num_users, per_topic, user_5_df, user_4_df)
  print(recommendations)
  calc_personalization(recommendations, num_movies, num_users)
  mapk(recommendations, num_users, user_5_df, user_4_df)

if not args.just_eval:
  if args.regen_words_df:
    if args.regen_data_lemmatized: # lemmatize all movies
      df = pd.read_csv("ml_netflix.csv").drop_duplicates('title')

      # remove non-english words. Reference: https://datascience.stackexchange.com/questions/46705/to-remove-chinese-characters-as-features
      df['description'] = df['description'].map(lambda x: re.sub("([^\x00-\x7F])+","", x))

      def sent_to_words(sentences):
          for sentence in sentences:
              yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations and special characters

      data_words = list(sent_to_words(df['description']))

      stop_words = stopwords.words('english')

      def remove_stopwords(texts):
          return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

      # Build the bigram
      bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10) # higher threshold fewer phrases.
      bigram_mod = gensim.models.phrases.Phraser(bigram)

      def make_bigrams(texts):
          return [bigram_mod[doc] for doc in texts]

      def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
          """https://spacy.io/api/annotation"""
          texts_out = []
          for sent in texts:
              doc = nlp(" ".join(sent))
              texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
          return texts_out

      # Remove Stop Words
      data_words_nostops = remove_stopwords(data_words)

      # Form Bigrams
      data_words_bigrams = make_bigrams(data_words_nostops)

      # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
      nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

      # Do lemmatization keeping only noun, adj, vb, adv
      data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

      if not os.path.isdir("cache"):
        os.mkdir("cache")
      with open("cache/data_lemmatized.txt", "w+") as f:
        f.write(str(data_lemmatized))

    else:
      with open("cache/data_lemmatized.txt", "r") as f:
        data_lemmatized = ast.literal_eval(f.read())

    word_to_id = {}
    all_words = []
    # need word to unique id
    # and num instances of each word
    # want table like with cols: word, word_id, doc_id
    word_id = 1
    for doc_id, doc in enumerate(data_lemmatized[0:NUM_MOVIES]):
      for word in doc:
        all_words.append([word_id, doc_id+1])

        if word not in word_to_id:
          word_to_id[word] = word_id
          word_id += 1

    words_df = pd.DataFrame(all_words, columns=['word_id', 'doc_id'])
    words_df.to_csv('cache/words_df.csv')
  else:
    print("Using cached vals")
    words_df = pd.read_csv('cache/words_df.csv')
    words_df = words_df[words_df['doc_id'] < NUM_MOVIES]

  print("Finish processing words")

  v = words_df.groupby('word_id').ngroups
  data = {"K": NUM_TOPICS, # num topics
                  "V": v, # num unique words
                  "M": NUM_MOVIES, # num movies
                  "N": len(words_df), # num total words
                  "w": words_df['word_id'].to_list(), # word ID for word n
                  "doc": words_df['doc_id'].to_list(), # doc ID for word n
                  "alpha": [1] * NUM_TOPICS, # topic prior
                  "beta": [1] * v # word prior
                }

  posterior = stan.build(program, data=data, random_seed=1)

  fit = posterior.sample(num_chains=1, num_samples=5)

  # Extracting traces
  movie_probs = fit["theta"]  # array with shape (NUM_MOVIES, NUM_TOPICS, NUM_SAMPLES)

  if not os.path.isdir("results"):
    os.mkdir("results")
  with open('results/theta.npy', 'wb') as f:
    np.save(f, movie_probs)
else:
  movie_probs = np.load('results/theta.npy') # read posterior vals from file

# EVALUATION
eval(movie_probs)