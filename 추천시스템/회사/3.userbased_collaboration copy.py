import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

 
# user-item 테이블 필요
# pivot table이용하여 만들기
data = pd.read_csv('score2.csv')
data = data.pivot_table('score', index='userid', columns='movieCd')
data

# 유저 정보, 영화정보 통합
info = pd.read_csv('C:/kwon/juno1412/추천시스템/movie_info_result.csv')
score = pd.read_csv('C:/kwon/score2.csv')
score = score[['userid', 'score']]

movie_data = pd.concat([score, info], axis=1, ignore_index=False)
movie_data.columns

# user-item 테이블 수정
data2 = movie_data.pivot_table('score', index='userid', columns='movieNm').fillna(0)
data2.shape

data2 = data2.transpose()

sim = cosine_similarity(data2, data2)
sim_df = pd.DataFrame(data = sim, index = data2.index, columns = data2.index)

# 영화의 유사도 50개
sim_df["힛쳐"].sort_values(ascending=False)[1:50] # 내림차순 정렬

# 영화 추천



