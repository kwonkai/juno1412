import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tables import Column
 
# user-item 테이블 필요
# pivot table이용하여 만들기
data = pd.read_csv('score2.csv')
data = data.pivot_table('score', index='userid', columns='movieCd')
data

# 유저 정보, 영화정보 통합
info = pd.read_csv('C:/kwon/juno1412/추천시스템/movie_info_result.csv')
score = pd.read_csv('C:/kwon/score1.csv')
score = score[['userid', 'score']]

movie_data = pd.concat([score, info], axis=1, ignore_index=False)

# user-item 테이블 수정
data2 = movie_data.pivot_table('score', index='userid', columns='movieNm').fillna(0)
data2.shape

data2 = data2.transpose()

sim = cosine_similarity(data2, data2)
sim_df = pd.DataFrame(data = sim, index = data2.index, columns = data2.index)

# 영화의 유사도 50개
sim_df["힛쳐"].sort_values(ascending=False)[1:50] # 내림차순 정렬

# 영화 추천

# 개인최적화 영화 추천
# score_arr : user * item, sim_arr : item * item
def predict_score(score_arr, sim_arr):
    sum_sr = score_arr @ sim_arr # 행렬곱 2x2 = .dot # 유저평가, 아이템 유사도 행렬곱
    sum_s_abs = np.array( [ np.abs(sim_arr).sum(axis=1)] ) # sim_arr 절댓값의 합 1차원 배열

    predict = sum_sr / sum_s_abs
    return predict

#성능평가

predict = predict_score(data.values, sim_df.values)
predict_metrix = pd.DataFrame(data=predict_score, index = data2.columns, columns = data2.index)
predict_metrix.head(3)





