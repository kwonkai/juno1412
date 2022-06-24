# -*- coding: utf-8 -*-
'''Netfilx Recommendation System : User database Collaborative Filtering'''

# 라이브러리 설정
import pandas as pd
import numpy as np
import math
import re

import matplotlib.pyplot as plt
#import seaborn as sns

from scipy.sparse import csr_matrix
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate # cross_validate = evaluate -> surprise 버전 변경으로 변경됨
#sns.set_style("ticks")

from sklearn.metrics.pairwise import cosine_similarity

# 데이터셋 가져오기

user_data1 = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/combined_data_1.txt', header = None, names=['user_id', 'Rating'], usecols = [0,1])

user_data1['Rating'] = user_data1['Rating'].astype(float)

print('User Dataset 1 shape : {}'.format(user_data1.shape))
print('-User Dataset sample-')
print(user_data1.iloc[::300000, :])

# 데이터 시각화
# agg() : 모든 열에 함수를 매핑
data_view = user_data1.groupby('Rating')['Rating'].agg(['count'])
data_view

# movie, user, rating count
# nunique() = 고유값 구하기
movie_count = user_data1.isnull().sum()[1]

user_count = user_data1['user_id'].nunique() - movie_count

rating_count = user_data1['user_id'].count() - movie_count

print(movie_count)
print(user_count)
print(rating_count)

# 평점 분포도 시각화
ax = data_view.plot(kind='barh', legend = False, figsize = (15,10))
plt.title('Total : {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, user_count, rating_count), fontsize=20)
plt.axis('off')

for i in range(1,6):
    ax.text(data_view.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f}%'.format(i, data_view.iloc[i-1][0]*100 / data_view.sum()[0]), color = 'white', weight = 'bold')

# 데이터 전처리

data_nan = pd.DataFrame(pd.isnull(user_data1.Rating)) # 
data_nan = data_nan[data_nan['Rating'] == True]
data_nan = data_nan.reset_index()
print(data_nan)

movie_np = []
movie_id = 1

# data_nan check
a = data_nan['index'][1:]
b = data_nan['index'][:-1] # 마지막 값 빼고 들고옴 4499 -> 추가 보충 필요
len(user_data1) - data_nan.iloc[-1, 0] - 1

for i,j in zip(data_nan['index'][1:], data_nan['index'][:-1]):
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

# user_data1 index 0~4498 # movie_np index 1~4498 -> 4499 index 추가 필요
record = np.full((1,len(user_data1) - data_nan.iloc[-1, 0] - 1), movie_id) # 428 # 4499번째 개수 맞음을 확인함
print(record)
movie_np = np.append(movie_np, record) # 
print(movie_np)

print('Movie numpy: {}'.format(movie_np))
print('Length: {}'.format(len(movie_np)))

# remove those Movie ID rows
# 결측치가 있는 row 제거
df = user_data1[pd.notnull(user_data1['Rating'])]

# 컬럼 = movie_id 추가
df['movie_id'] = movie_np.astype(int)
df['user_id'] = df['user_id'].astype(int)
print('-Dataset examples-')
print(df.iloc[::5000000, :])



## 데이터 유효값 전처리
# 1) 리뷰가 너무 적은 영화 제거
# 2) 리뷰를 너무 적게 제공하는 고객 제거

number = ['count', 'mean']


# movie/user 최소 유효값 구하기
# movie_summary = index.movie_id, columns = count, mean
movie_summary = df.groupby('movie_id')['Rating'].agg(number)
movie_summary.index = movie_summary.index.map(int)


# 분위값 구한 뒤 반올림하기
movie_mark = round(movie_summary['count'].quantile(0.8),0) 
remove_movie_list = movie_summary[movie_summary['count'] < movie_mark].index
len(remove_movie_list)
print('Movies minimum times of review: {}'.format(movie_mark))


# user 최소 유효값 구하기
user_summary = df.groupby('user_id')['Rating'].agg(number)
user_summary.index = user_summary.index.map(int)
  

user_mark = round(user_summary['count'].quantile(0.8),0) # quantile 분위값 구하기
remove_user_list = user_summary[user_summary['count'] < user_mark].index
len(remove_user_list)
print('Users minimum times of review: {}'.format(user_mark))


# 데이터 소거
# remove user/movie list만 가져오기
# 유효 기준치를 통과한 데이터
df = df[~df['movie_id'].isin(remove_movie_list)]
df = df[~df['user_id'].isin(remove_user_list)]
print('-Data Examples-')
print(df.iloc[::5000000, :])

## 데이터 셋 피벗테이블 정리
df_pivot = pd.pivot_table(df,values = 'Rating', index = 'user_id', columns = 'movie_id')
print(df_pivot.shape)

## 데이터 mapping
movie_title = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/movie_titles.csv', encoding = 'ISO-8859-1', header = None, names = ['movie_id', 'year', 'name'])

movie_title.set_index('movie_id', inplace = True)
print(movie_title.head(10))

# index (0, 1, 2) # rating_scale =(1, 5)
# Reader() # user ; item ; rating ; [timestamp]
reader = Reader() 
reader = Reader(rating_scale=(1,5))  # parameter error
# 10만개 데이터만 가져오기 -> 속도
# reader 기준으로 데이터를 파싱
# 학습용 데이터
m_data = Dataset.load_from_df(df[['user_id', 'movie_id', 'Rating']][:100000], reader)

#  svd() + crossvalidate(algo, m_data, loss, fold = 5
algo = SVD()
cross_validate(algo, m_data, measures=['RMSE', 'MAE'], cv=5, verbose=True) # ??? cross_validation why? # just loss??


# just check
# 
df_svd = df[(df['user_id'] == 4499) & (df['Rating'] == 5)]
df_svd = df_svd.set_index('movie_id')
df_svd = df_svd.join(movie_title)['name']
print(df)
print(movie_title)
print(df_svd)
# print(df_svd)

# user_svd = movie_title raw data -> 유효값 필터링
user_svd = movie_title.copy()
user_svd = user_svd.reset_index()
user_svd = user_svd[~user_svd['movie_id'].isin(remove_movie_list)]

# 코드분석 필요
user_data = Dataset.load_from_df(df[['user_id', 'movie_id', 'Rating']], reader)
#전체데이터 학습이용
trainset = user_data.build_full_trainset()
algo.fit(trainset)

print(user_svd)

algo.predict(1, 2314, 2) # predict(user_id, movie_id, real_rating, predict_rating)

## surprise 라이브러리에서
# predict = uid, iid, r_id(실제 평가), est(예측 평가)
user_svd['Prediction'] = user_svd['movie_id'].apply(lambda x: algo.predict(785314, x).est)

user_svd = user_svd.drop('movie_id', axis = 1)

user_svd = user_svd.sort_values('Prediction', ascending=False)
print(user_svd.head(10))

# 예측 점수 추천시스템
# 분석 # ??

## 데이터 mapping
# movie_titles.csv 가져오기
movie_title = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/movie_titles.csv', encoding = 'ISO-8859-1', header = None, names = ['movie_id', 'year', 'name'])
movie_title.set_index('movie_id', inplace = True)

def recommend(title, min_count): # movie_title = title, min_count 
    print("For movie ({})".format(movie_title))
    print("- Top 10 movies recommended based on Pearsons'R correlation - ")
    i = int(movie_title.index[movie_title['name'] == title][0]) 
    target = df_pivot[i] # pivot_table target
    similar_to_target = df_pivot.corrwith(target) # pearson R similarity = target_movie / every movie # corrwith()모든 변수간 상관계수
    corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])
    corr_target.dropna(inplace = True) # 결측치 제거
    corr_target = corr_target.sort_values('PearsonR', ascending = False)
    corr_target.index = corr_target.index.map(int)
    corr_target = corr_target.join(movie_title).join(movie_summary)[['PearsonR', 'name', 'count', 'mean']]
    print(corr_target[corr_target['count']>min_count][:10].to_string(index=False))

recommend("Batman Begins", 0)

recommend("Something's Gotta Give", 0)

recommend("X2: X-Men United", 0)


def recommend(title, min_count): # movie_title = title, min_count 
    print("For movie ({})".format(movie_title))
    print("- Top 10 movies recommended based on Pearsons'R correlation - ")
    i = int(movie_title.index[movie_title['name'] == title][0]) 
    target = df_pivot[i] # pivot_table target
    other_user = df_pivot[df_pivot.index != target]
    # other_user = df_pivot[~df_pivot.isin(target)]/
    sim_co = cosine_similarity(target, other_user)
    corr_target = pd.DataFrame(sim_co, columns = ['cosine_similarity'])
    corr_target.dropna(inplace = True) # 결측치 제거
    corr_target = corr_target.sort_values('cosine_similarity', ascending = False)
    corr_target.index = corr_target.index.map(int)
    corr_target = corr_target.join(movie_title).join(movie_summary)[['PearsonR', 'name', 'count', 'mean']]
    print(corr_target[corr_target['count']>min_count][:10].to_string(index=False))



recommend("Batman Begins", 0)

recommend("Something's Gotta Give", 0)

recommend("X2: X-Men United", 0)