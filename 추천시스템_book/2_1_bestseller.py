# Created or modified on May 2022
# Author: 임일
# Best-seller 추천

import numpy as np
import pandas as pd

# u.user 파일을 DataFrame으로 읽기 
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_reco_book/u.user', sep='|', names=u_cols, encoding='latin-1')
users = users.set_index('user_id')
users.head()

# u.item 파일을 DataFrame으로 읽기
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_reco_book/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies.set_index('movie_id')
movies.head()

# u.data 파일을 DataFrame으로 읽기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_reco_book/u.data', sep='\t', names=r_cols, encoding='latin-1') 
ratings = ratings.set_index('user_id')
ratings['rating'] = ratings['rating'].astype(float)



# best_seller 추천
def recom_book(items):
    movie_sort = movie_mean.sort_values(ascending=False)[:items]
    recom_movies = movies.loc[movie_sort.index]
    recommendations = recom_movies['title']
    return recommendations

def recom_movie2(n_items):
   return movies.loc[movie_mean.sort_values(ascending=False)[:n_items].index]['title']


movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
a = recom_book(5)
print(a)


# 모델 정확도 계산하기
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

rmse = []
for user in set(ratings.index):
    y_true = ratings.loc[user]['rating']
    y_pred = movie_mean[ratings.loc[user]['movie_id']]
    accuracy = RMSE(y_true, y_pred)
    rmse.append(accuracy)
print(np.mean(rmse))