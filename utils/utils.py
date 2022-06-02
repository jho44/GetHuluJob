import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_by_storyline(title, df): # credit: https://github.com/nicoleeesim/content-based_movie_recommender
    '''
    gets top 10 recs for a given movie title
    '''
    recommended = []
    top10_list = []

    title = title.lower()
    df['title'] = df['title'].str.lower()
    topic_num = df[df['title']==title].Topic.values
    if len(topic_num) == 0:
        print(title + " not in database")
        return
    doc_num = df[df['title']==title].Doc.values

    output_df = df[df['Topic']==topic_num[0]].sort_values('Probability', ascending=False).reset_index(drop=True)

    index = output_df[output_df['Doc']==doc_num[0]].index[0]

    # return the 10 results with closest probability of belonging to topic `topic_num`
    top10_list += list(output_df.iloc[index-5:index].index)
    top10_list += list(output_df.iloc[index+1:index+6].index)

    output_df['title'] = output_df['title'].str.title()
    for each in top10_list:
        recommended.append(output_df.iloc[each].title)

    return recommended

def get_helper_vals(users_df):
    num_movies = len(users_df.drop_duplicates('title'))
    num_users = users_df['userId'].nunique()
    return num_movies, num_users

def sort_probs(num_topics, df): # nikki's
    '''
    df: dataframe with key (title, topic, topic_probability)
    '''
    # for each topic, precompute sorted probabilities
    per_topic = [] # list of dfs

    for topic in range(0, num_topics):
        output_df = df[df['Topic']==topic].sort_values('Probability', ascending=False).reset_index(drop=True)
        per_topic.append(output_df)

    return per_topic

def reassign_ids(df, property): # nikki's
    codes, uniques = pd.factorize(df[property], sort=True)
    df[property] = codes
    return df

def get_users_faves(users_df):
     # find all 4 and 5 star ratings
    user_5_df = users_df[users_df['rating']==5]
    user_4_df = users_df[users_df['rating']==4]
    return user_5_df, user_4_df

def get_recs_for_all_users(num_users, per_topic, user_5_df, user_4_df): # nikki's
    # calculate recommendations for each user
    """
    users_df: dataframe with ratings for each movie -- userIds should be reassigned!

    as suggested in project proposal, number of recommendations k depends on user
    we set k = minimum(number of movies the user has given 5 stars, number of movies the user has given 4 stars)

    we calculate recommendation by taking all movies that the user rated 5 stars (or 4 stars if no 5 stars)
    then, for each movie, we find the topic it is in by finding the topic that it has the highest probability in
    and find the movie in that topic with probability closest to that movie's topic probabiity
    """

    # for each user, calculate recommended movies
    # recommendations is 2d array where recommendations[i] is a list with first value = userId,
    # subsequent values are k recommended titles for the user
    # where k is number of 5 star ratings given by user and if no 5 star ratings, number of 4 star ratings given by user
    recommendations = []
    for user in range(num_users):
        # find number of 5 ratings
        temp_df = user_5_df[user_5_df['userId']==user]
        temp_df
        # if no 5 ratings, find number of 4 ratings
        if temp_df.empty:
            temp_df = user_4_df[user_4_df['userId']==user]
        # # if no 4 ratings, go to next user
        # if temp_df.empty:
        #     continue
        recommendations.append([user])
        # for each movie
        for index, row in temp_df.iterrows():
            topic = row['Topic']
            movieId = row['movieId']
            prob = row['Probability']
            topic_df = per_topic[topic]
            # find closest probability in topic to the current movie
            index = topic_df[topic_df['movieId']==movieId].index[0]
            # accounting for out of bound accessing
            if index == 0:
                row_below = topic_df.iloc[index+1]
                recommendations[-1].append(row_below['movieId'])
                continue
            if index == len(topic_df)-1:
                row_above = topic_df.iloc[index-1]
                recommendations[-1].append(row_above['movieId'])
                continue
            row_above = topic_df.iloc[index-1]
            row_below = topic_df.iloc[index+1]
            if row_below['Probability'] - prob < row_above['Probability'] - prob:
                recommendations[-1].append(row_below['movieId'])
            else:
                recommendations[-1].append(row_above['movieId'])

    return recommendations

# EVAL FUNCS
def calc_personalization(recommendations, num_movies, num_users): # nikki's
    # calculate personalization matrix
    # https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093
    # create num_user x num_movies
    personalization = np.zeros((num_users, num_movies))

    for row in recommendations:
        user = row[0]
        for movie_ind in row[1:]:
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

def mapk(recommendations, num_users, user_5_df, user_4_df): # nikki's
    # calculate MAP @ k
    # http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html

    # for each user
    map_k = 0
    for user in range(num_users):
        l = len(recommendations[user]) - 1
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
            user_rec = recommendations[user][1:k+1]
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
    map_k /= float(num_users)
    print(f'MAP@K: {map_k}')