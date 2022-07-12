# Created or modified on May 2022
# Author: 임일
# 클러스터별 Best-seller 추천

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# u.user 파일을 DataFrame으로 읽기 
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_reco_book/u.user', sep='|', names=u_cols, encoding='latin-1')

# u.item 파일을 DataFrame으로 읽기
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_reco_book/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies[['movie_id', 'title']]

# u.data 파일을 DataFrame으로 읽기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_reco_book/u.data', sep='\t', names=r_cols, encoding='latin-1') 
ratings = ratings.drop('timestamp', axis=1)


# train_test_split
x = ratings.copy()
y = ratings['user_id']

x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.25, stratify=y)

# 모델 정확도 RMSE 계산
# (y_true 값 - y_pred)^2의 평균 후 제곱근
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

# 모델별로 rmse 계산하기


