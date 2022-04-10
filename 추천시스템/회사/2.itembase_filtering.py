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
 
# user-item 테이블 필요
# pivot table이용하여 만들기
data = pd.read_csv('추천시스템/score2.csv')
data = data.pivot_table('score', index='userid', columns='movieCd')


# 유저 정보, 영화정보 통합
info = pd.read_csv('추천시스템/movie_info_result.csv')
info = info[['movieCd', 'movieNm', 'genreNm','directors_ko']]
score = pd.read_csv('추천시스템/score2.csv')
score = score[['score', 'userid']]
movie_data = pd.concat([info, score], axis=1)
movie_data

# user-item 테이블 수정
data_t = movie_data.pivot_table('score', index='userid', columns='movieNm').fillna(0)
data_t = data_t.transpose()


sim = cosine_similarity(data_t, data_t)
sim_df = pd.DataFrame(data = sim, index = data_t.index, columns = data_t.index)
sim_df

# 영화의 유사도 50개
sim_df["스파이더맨: 노 웨이 홈"].sort_values(ascending=False)[1:50] # 내림차순 정렬

# 영화 추천

# 개인최적화 영화 추천
# score_arr : user * item, sim_arr : item * item
def predict_score(score_arr, sim_arr):
    ratings_pred = score_arr.dot(sim_arr)/ np.array([np.abs(sim_arr).sum(axis=1)]) 
    return ratings_pred
    # predict = score_arr.dot(sim_arr) / np.array([np.abs(sim_arr).sum(axis=1)])
    # sum_sr = score_arr @ sim_arr # 행렬곱 2x2 = .dot # 유저평가, 아이템 유사도 행렬곱
    # sum_s_abs = np.array( [ np.abs(sim_arr).sum(axis=1)] ) # sim_arr 절댓값의 합 1차원 배열

    # predict = sum_sr / sum_s_abs
    # score_arr
    # return predict

#성능평가
data_t.shape
sim_df.shape

predict = predict_score(data_t.values, sim_df.values)
predict_metrix = pd.DataFrame(data=predict_score, index = data_t.columns, columns = data_t.index)
predict_metrix.head(3)

from sklearn.metrics import mean_squared_error

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    
    return mean_squared_error(pred, actual)

MSE = get_mse(predict, predict_metrix.values)
print(f'인접 이웃: {MSE:.4f}')