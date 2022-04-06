# import pandas as pd 
# import numpy as np 

# movie = pd.read_csv('C:/kwon/juno1412/추천시스템/movies.csv') 
# score = pd.read_csv('C:/kwon/juno1412/추천시스템/ratings.csv')
# score = score[['userId', 'movieId', 'rating']]
# score_t = score.pivot_table('rating', index = 'userId', columns= 'movieId') # 행열 반대로

# score_movie = pd.merge(score, movie, on="movieId")
# score_t = score_movie.pivot_table('rating', index = 'userId', columns= 'title') # 행열 반대로
# score_t = score_t.fillna(0)

# score_t_m = score_t.transpose()

# from sklearn.metrics.pairwise import cosine_similarity
# movie_sim = cosine_similarity(score_t_m, score_t_m)
# movie_sim_df = pd.DataFrame(data=movie_sim, index = score_t.columns, columns= score_t.index)


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tables import Column
 
# user-item 테이블 필요
# pivot table이용하여 만들기
data = pd.read_csv('C:/kwon/juno1412/추천시스템/ratings.csv')
data = data.pivot_table('rating', index='userId', columns='movieId')
data

# 유저 정보, 영화정보 통합
info = pd.read_csv('C:/kwon/juno1412/추천시스템/movies.csv')
movie_data = pd.merge(data,info, on='movieId')

# user-item 테이블 수정
data2 = movie_data.pivot_table('rating', index='userId', columns='title').fillna(0)
data2 = data2.transpose()
data2

sim = cosine_similarity(data2, data2)
sim_df = pd.DataFrame(data = sim, index = data2.index, columns = data2.index)
sim_df

# 영화의 유사도 50개
sim_df["Nadja (1994)"].sort_values(ascending=False)[1:50] # 내림차순 정렬

# 영화 추천

# 개인최적화 영화 추천
# score_arr : user * item, sim_arr : item * item
def predict_score(score_arr, sim_arr):
    sum_sr = score_arr @ sim_arr # 행렬곱 2x2 = .dot # 유저평가, 아이템 유사도 행렬곱
    sum_s_abs = np.array( [ np.abs(sim_arr).sum(axis=1)] ) # sim_arr 절댓값의 합 1차원 배열

    predict = sum_sr / sum_s_abs
    return predict

#성능평가

predict = predict_score(data2.values, sim_df.values)
predict_metrix = pd.DataFrame(data=predict_score, index = data2.columns, columns = data2.index)
predict_metrix.head(3)

from sklearn.metrics import mean_squared_error

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    
    return mean_squared_error(pred, actual)

MSE1 = get_mse(predict_score, data2.values)
print(f'아이템 기반 모든 인접 이웃 MSE: {MSE1:.4f}')