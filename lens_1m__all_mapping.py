# 1. 라이브러리 설정
import pandas as pd
import numpy as np

# 시드값 고정
import tensorflow as tf
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# 2. 데이터 불러오기
ml_movie = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/y_class_콘텐츠 메타데이터/movie_lens_official/ml-1m/ml-1m/movies.dat', sep = '::', header = None, encoding = 'latin-1')
ml_movie.columns = ['movieId', 'title', 'genres']

rating = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/y_class_콘텐츠 메타데이터/movie_lens_official/ml-1m/ml-1m/ratings.dat', sep='::',header = None, engine='python',encoding='latin-1').drop(columns=[3], axis=1)
rating.columns = ['userId', 'movieId', 'rating']

# ml_movie['movieId'] vs rating['movieId'] 비교 후, rating 0개 콘텐츠 뽑아내기
rating_list = rating['movieId'].tolist()
except_movie_df = ml_movie[~ml_movie['movieId'].isin(rating_list)]

# rating == 0인 콘텐츠 list만들기
except_movie_list = except_movie_df['movieId'].tolist()

# rating['movieId']에 없는지 다시 한번 확인
# rating_except_check = rating[rating['movieId'].isin(except_movie_list)]

# ml_movie에서 rating 0인 콘텐츠 제거
ml_movie = ml_movie[~ml_movie['movieId'].isin(except_movie_list)]



# 2. 장르 & 사용자정보 붙이기
# 장르전처리
ml_movie_genre= ml_movie.copy()
ml_movie_genre['genres'] = ml_movie_genre.genres.str.split('|')


ml_movie_genre = pd.DataFrame(ml_movie_genre['genres'].to_list())

ml_movie_genre.columns = ['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6']

# genre1 join              
ml_movie = ml_movie.reset_index()
ml_movie_df = ml_movie.join(ml_movie_genre['genre1']).drop(columns=['index', 'genres'], axis=1).reset_index()

# 장르개수 확인
print(ml_movie_df['genre1'].value_counts())


# 유저정보 붙이기
user = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/y_class_콘텐츠 메타데이터/movie_lens_official/ml-1m/ml-1m/users.dat', sep='::',header = None, engine='python',encoding='latin-1').drop(columns=[4], axis=1)
user.columns = ['userId', 'gender', 'age', 'occupation']
ml_movie_df = pd.merge(ml_movie_df,user, on='userId' )


# 3. count 컬럼 생성하기
rating = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/y_class_콘텐츠 메타데이터/movie_lens_official/ml-1m/ml-1m/ratings.dat', sep='::',header = None, engine='python',encoding='latin-1').drop(columns=[3], axis=1)
rating.columns = ['userId', 'movieId', 'rating']
print(rating['movieId'].nunique())

# movieId 컬럼 복사 = movieId2
rating['movieId2'] = rating['movieId']
rating['movieId2'] = rating['movieId']

# movieId를 인덱스로 변경
# rating_df = movieId가 인덱스인 rating dataframe
rating_df = rating.set_index('movieId')

# count 컬럼만들기
# movieId, count 컬럼의 dataframe 생성
count_list = pd.DataFrame(rating_df['movieId2'].value_counts()).reset_index()
count_list.columns = ['movieId','count']

# ml_movie_df와 count_list 합치기
ml_movie_df_all = pd.merge(ml_movie_df, count_list, on='movieId')
# ml_movie_df_count = pd.merge(count_list, rating, on='movieId').drop(columns=['movieId2'], axis=1)


ml_movie_df_sort = ml_movie_df_all.sort_values(by="count", ascending=False).reset_index().drop(columns=['level_0','index'], axis=1)
ml_movie_df_sort = ml_movie_df_sort.sample(frac=1)
ml_movie_df_sort.to_csv('/home/juno/workspace/user_collaborative_filtering/deeplearning_recomend/yclass/lens_all_mapping.csv')


