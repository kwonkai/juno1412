﻿import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

movie_data = pd.read_csv('C:\kwon\juno1412\추천시스템\movie_info_result.csv')
movie_data

# 데이터 전처리
# shortContent, genreNm, directNm별로 전처리
movie_content = movie_data[['movieCd', 'movieNm', 'actors_ko', 'genreNm', 'directors_ko']]
movie_content = movie_content[['movieCd', 'movieNm', 'actors_ko', 'genreNm', 'directors_ko']].fillna("")

# content 컬럼 추가
movie_content['content'] = movie_content["actors_ko"] + movie_content["genreNm"] + movie_content["directors_ko"]
movie_content.head()

# 컬럼별 ','으로 구분하여 통합시키기
colums = ['actors_ko', 'genreNm', 'directors_ko']
movie_content['content'] = movie_content[colums].apply(lambda row: ','.join(row.values), axis=1)


# 4. 문자열 genreNm 칼럼을 Count 기반으로 피처 벡터화 변환
# CountVectorizer = 단어 벡터화 카운트 정렬
# TfidfTransformer = TF-IDF로 단어 벡터화
# TF-IDF : 문서 내 단어빈도를 이용한 단어들마다 중요한 정도를 가중치주는 방법. 유사도 구하기/검색 등에 사용됨
# TF-IDF = TF * IDF, TF(특정 문서 d에서 특정단어 t 등장횟수), 

count_vect = CountVectorizer(min_df=0, ngram_range=(1,1))  #단어 출현횟수 카운팅
merge_content_c = count_vect.fit_transform(movie_content['content']) # train_dataset에만 사용
# tfidf_transformer = TfidfTransformer()
# tfidf_vectorizer = TfidfVectorizer(min_df=0, ngram_range=(1.1)) # 토큰빈도 1이상, (min_n, max_n) 튜플

tfidf_transformer = TfidfTransformer()
merge_content = tfidf_transformer.fit_transform(merge_content_c)

# 문서, 단어(46919, 53990)
merge_content_c.shape

# 5. 리스트 객체 문자열 변경, count 피처 벡터화하기
director_content_sim = cosine_similarity(merge_content_c, merge_content_c)
director_content_sim[0]

# 6. 특정영화 정보, 유사도 뽑아내기
def find_genre_score(movie_content, sim_metrics, title_name, top_n=100):

    # 입력한 영화 인덱스
    movieNm = movie_content[movie_content['movieNm'] == title_name]
    movieNm_index = movieNm.index.values

    # 입력한 영화 유사도 데이터 프레임을 추가하기
    movie_content["similarity"] = sim_metrics[movieNm_index, :].reshape(-1,1)

    # 유사도 내림차순 정렬(상위 인덱스 100개 추출)
    temp = movie_content.sort_values(by="similarity", ascending=False)
    final_index = temp.index.values[ : top_n]

    return movie_content.iloc[final_index]

# 7. 특정영화 유사한 장르 영화를 평점 순위별로 추천하기

# 7-1. 해리포터와 높은 유사도를 가지는 영화 순위별 정렬하기
similar_movie = find_genre_score(movie_content, director_content_sim, '해리포터와 비밀의 방', 30)
similar_movie = similar_movie[['movieCd', 'movieNm', 'similarity', 'content']]
similar_movie