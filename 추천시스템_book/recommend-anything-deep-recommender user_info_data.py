# To store the data
from pickletools import float8
import pandas as pd

# To do linear algebra
import numpy as np

import statistics

# To create plots
import matplotlib.pyplot as plt

# To create interactive plots
# from plotly.offline import init_notebook_mode, plot, iplot
# import plotly.graph_objs as go
# init_notebook_mode(connected=True)

# To shift lists
from collections import deque

# To compute similarities between vectors
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# To use recommender systems
import surprise as sp
from surprise.model_selection import cross_validate

# To create deep learning models
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model

# To create sparse matrices
from scipy.sparse import coo_matrix

# To light fm
# from lightfm import LightFM
# from lightfm.evaluation import precision_at_k

# To stack sparse matrices
from scipy.sparse import vstack


### 1. movielens 데이터 파일 불러오기
# u.user 파일을 DataFrame으로 읽기 
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_reco_book/u.user', sep='|', names=u_cols, encoding='latin-1')
users['user_id'] = users['user_id'].astype(int)

# u.item 파일을 DataFrame으로 읽기
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_reco_book/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies[['movie_id', 'title']]

# u.data 파일을 DataFrame으로 읽기
# user_rating 전처리 : timestamp 제거, rating 내부의 rating 값 이상의 데이터 제거
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_reco_book/u.data', sep='\t', names=r_cols, encoding='latin-1') 
ratings = ratings.drop('timestamp', axis=1).dropna()
ratings[['user_id', 'rating']] = ratings[['user_id', 'rating']].astype(float).astype(int)
ratings = ratings[ratings.rating < 6]
ratings = ratings[ratings.rating > -1]
user_metadata = pd.merge(users, ratings, on='user_id')
print(user_metadata)


### 2.유효성 검사
user_metadata_userrating = user_metadata.groupby('user_id')['rating'].count()
statistics.mean(user_metadata_userrating)
user_metadata_contentrating = user_metadata.groupby('movie_id')['rating'].count()
statistics.mean(user_metadata_contentrating)

# Filter sparse movies
min_movie_ratings = 50
filter_movies = (user_metadata['movie_id'].value_counts()>min_movie_ratings)
filter_movies = filter_movies[filter_movies].index.tolist()

# Filter sparse users
min_user_ratings = 10
filter_users = (user_metadata['user_id'].value_counts()>min_user_ratings)
filter_users = filter_users[filter_users].index.tolist()

# Actual filtering
df_filterd = user_metadata[(user_metadata['movie_id'].isin(filter_movies)) & (user_metadata['user_id'].isin(filter_users))]
del filter_movies, filter_users, min_movie_ratings, min_user_ratings
print('Shape User-Ratings unfiltered:\t{}'.format(user_metadata.shape))
print('Shape User-Ratings filtered:\t{}'.format(df_filterd.shape))



## 데이터 섞기 -> 순서대로 되어있는 데이터 섞어 train_test 잘 섞이게 함
df_filterd = df_filterd.drop('zip_code', axis=1).sample(frac=1).reset_index(drop=True)

n = 6700

df_filterd_train = df_filterd[:10000]
df_filterd_test = df_filterd[-n:]



# Create user- & movie-id mapping
user_id_mapping = {id:i for i, id in enumerate(user_metadata['user_id'].unique())}
movie_id_mapping = {id:i for i, id in enumerate(user_metadata['movie_id'].unique())}

# Use mapping to get better ids
user_metadata['user_id'] = user_metadata['user_id'].map(user_id_mapping)
user_metadata['movie_id'] = user_metadata['movie_id'].map(movie_id_mapping)


##### Combine both datasets to get movies with metadata
# Preprocess metadata
# users : user_id age sex occupation zip_code
user_info = users.copy()
user_info = user_info.drop('zip_code', axis=1)

# user 정보 나누기 : 나이 성별 직업
user_age = user_info[['user_id', 'age']].set_index('user_id')
user_sex = user_info[['user_id', 'sex']].set_index('user_id')
user_occ = user_info[['user_id', 'occupation']].set_index('user_id')


# tf_idf (문장-단어 벡터화) 행렬
# STOP_WORD =? 
tfidf = TfidfVectorizer()
# tfidf_hybrid_age = tfidf.fit_transform(user_age['age'])/
tfidf_hybrid_sex = tfidf.fit_transform(user_age['sex'])
tfidf_hybrid_occ = tfidf.fit_transform(user_age['occupation'])

mapping_age = {id:i for i, id in enumerate(tfidf_hybrid_age.index.unique())}
mapping_sex = {id:i for i, id in enumerate(tfidf_hybrid_sex.index.unique())}
mapping_occ = {id:i for i, id in enumerate(tfidf_hybrid_occ.index.unique())}


# df_hybrid_train shape : (300000, )
# tfidf_hybrid shape : (1, 24144)
train_tfidf = []
# Iterate over all movie-ids and save the tfidf-vector
for id in df_filterd_train[''].values:
    index = mapping[id]
    train_tfidf.append(tfidf_hybrid[index])
    
test_tfidf = []
# Iterate over all movie-ids and save the tfidf-vector
for id in df_hybrid_test['Movie'].values:
    index = mapping[id]
    test_tfidf.append(tfidf_hybrid[index])
print(test_tfidf)


# Stack the sparse matrices
# train_tfidf = train_tfidf.transfose()/
# train_tfidf = train_tfidf.stack()
# sparse 쌓기
# 1: [0,...,25] 2:[0,...35] -> 
train_tfidf = vstack(train_tfidf)
print(train_tfidf)
test_tfidf = vstack(test_tfidf)
train_tfidf = train_tfidf.toarray()
test_tfidf = test_tfidf.toarray()







# Filter all ratings with metadata
# df_id_descriptions : 영화 개요
# df에서 data 컬럼 삭제 후 movie 컬럼을 index로 설정 -> 4499개의 영화에 대한 user 평점 데이터를 순서대로 가지고 있음
# df_id_description의 'Id'(=movie_id)를 가져오고 overview 삭제
# 인덱스를 초기화하고 index 이름을 Movie로 다시 변경해줌
# df_hybrid = Movie User Rating의 열을 가진 880만개의 데이터로 전처리 됨 
df_hybrid1 = df.drop('Date', axis=1).set_index('Movie').join(df_id_descriptions).dropna().drop('overview', axis=1)
df_hybrid = df_hybrid1.reset_index().rename({'index':'Movie'}, axis=1)
print(df)
print(df_id_descriptions)
print(df_hybrid1)
print(df_hybrid)

# Split train- & testset
n = 100000
df_hybrid = df_hybrid.sample(frac=1).reset_index(drop=True)
df_hybrid_train = df_hybrid[:100000]
df_hybrid_test = df_hybrid[-n:]


# tf_idf (문장-단어 벡터화) 행렬
# STOP_WORD =? 
tfidf = TfidfVectorizer(stop_words='english')
print(tfidf)
tfidf_hybrid = tfidf.fit_transform(df_id_descriptions['overview'])
print(tfidf_hybrid)

# Get mapping from movie-ids to indices in tfidf-matrix
# idx와 movie_id를 mapping
# 결측치 찾기 + 결측치 숫자
# 중복값 685, 유일값 6254
# df_id_descriptions1 = df_id_descriptions.duplicated(keep='first').value_counts()/
# df_id_descriptions1 = pd.concat([df_id_descriptions, df_id_descriptions1], axis=1)


# mapping = movie_id, idx mapping, 
# 데이터 개수 6938개 -> 6104 : unique() 유일 값만 거르기
mapping = {id:i for i, id in enumerate(df_id_descriptions.index.unique())}
print(mapping)


# df_hybrid_train shape : (300000, )
# tfidf_hybrid shape : (1, 24144)
train_tfidf = []
# Iterate over all movie-ids and save the tfidf-vector
for id in df_hybrid_train['Movie'].values:
    index = mapping[id]
    train_tfidf.append(tfidf_hybrid[index])
    
test_tfidf = []
# Iterate over all movie-ids and save the tfidf-vector
for id in df_hybrid_test['Movie'].values:
    index = mapping[id]
    test_tfidf.append(tfidf_hybrid[index])
print(test_tfidf)


# Stack the sparse matrices
# train_tfidf = train_tfidf.transfose()/
# train_tfidf = train_tfidf.stack()
# sparse 쌓기
# 1: [0,...,25] 2:[0,...35] -> 
train_tfidf = vstack(train_tfidf)
print(train_tfidf)
test_tfidf = vstack(test_tfidf)
train_tfidf = train_tfidf.toarray()
test_tfidf = test_tfidf.toarray()




##### Setup the network
# Network variables
user_embed = 10
movie_embed = 10


# Create two input layers
user_id_input = Input(shape=[1], name='user')
movie_id_input = Input(shape=[1], name='movie')
tfidf_input = Input(shape=[24144], name='tfidf', sparse=True)

# Create separate embeddings for users and movies
user_embedding = Embedding(output_dim=user_embed,
                           input_dim=len(user_id_mapping),
                           input_length=1,
                           name='user_embedding')(user_id_input)
movie_embedding = Embedding(output_dim=movie_embed,
                            input_dim=len(movie_id_mapping),
                            input_length=1,
                            name='movie_embedding')(movie_id_input)

# Dimensionality reduction with Dense layers
tfidf_vectors = Dense(128, activation='relu')(tfidf_input)
tfidf_vectors = Dense(32, activation='relu')(tfidf_vectors)

# Reshape both embedding layers
user_vectors = Reshape([user_embed])(user_embedding)
movie_vectors = Reshape([movie_embed])(movie_embedding)

# Concatenate all layers into one vector
both = Concatenate()([user_vectors, movie_vectors, tfidf_vectors])

# Add dense layers for combinations and scalar output
dense = Dense(512, activation='relu')(both)
dense = Dropout(0.2)(dense)
output = Dense(1)(dense)


# Create and compile model
model = Model(inputs=[user_id_input, movie_id_input, tfidf_input], outputs=output)
model.compile(loss='mse', optimizer='adam')
model.summary()

# Train and test the network
model.fit([df_hybrid_train['User'], df_hybrid_train['Movie'], train_tfidf],
          df_hybrid_train['Rating'],
          batch_size=128,
          epochs=30,
          validation_split=0.2,
          shuffle=True,
          verbose=1)


model.save('/home/juno/workspace/user_collaborative_filtering/model_save/best_model_history/hybrid_deeplearning_movie3.h5')
# np.save('/home/juno/workspace/user_collaborative_filtering/model_save/best_model_history/hybrid_deeplearning_movie3.npy', model.history)


## 여기서 펑   
y_pred = model.predict([df_hybrid_test['User'], df_hybrid_test['Movie'], test_tfidf])
y_true = df_hybrid_test['Rating'].values

mse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False))
print('\n\nKeras Hybrid Deep Learning: {:.4f} MSE'.format(mse))
print('\n\nKeras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))

'''
### algorithm test
# Load dataset into surprise specific data-structure
data = sp.Dataset.load_from_df(df_filterd[['User', 'Movie', 'Rating']].sample(20000), sp.Reader())

benchmark = []
# Iterate over all algorithms
for algorithm in [sp.SVD(), sp.SVDpp(), sp.SlopeOne(), sp.NMF(), sp.NormalPredictor(), sp.KNNBaseline(), sp.KNNBasic(), sp.KNNWithMeans(), sp.KNNWithZScore(), sp.BaselineOnly(), sp.CoClustering()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    
    # Store data
    benchmark.append(tmp)
'''