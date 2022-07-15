# To store the data
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

# To stack sparse matrices
from scipy.sparse import vstack


# 2. 데이터 불러오기 
# anime & rating
anime = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/anime/anime.csv')
                        #       encoding = 'ISO-8859-1', 
                        #    header = None, 
                        #    names = ['anime_id','name','genre','type','episodes','rating','members'])                        
print('Shape Movie-Titles:\t{}'.format(anime.shape))

rating = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/anime/rating.csv')

# 3. 데이터 전처리
# 데이터 시각화
# 콘텐츠를 평가한 유저 숫자
rating_per_user = rating.groupby('user_id')['rating'].count()
statistics.mean(rating_per_user.tolist()) # 1인당 106.28765558049378

# 콘텐츠별 평가된 평점 등급
ratings_per_anime = rating.groupby('anime_id')['rating'].count()
statistics.mean(ratings_per_anime.tolist()) # 1콘텐츠당 697.3번 리뷰


# rating(평가된 콘텐츠 전처리)
# 너무 적게 평가된 콘텐츠 제거(50개 이하 평가 데이터 제거)
ratings_per_anime_df = pd.DataFrame(ratings_per_anime)
filtered_ratings_per_anime_df = ratings_per_anime_df[ratings_per_anime_df.rating >= 50]
popular_anime = filtered_ratings_per_anime_df.index.tolist()

# anime(콘텐츠를 평가한 유저 데이터 처리)
# 너무 적게 평가한 유저 제거 -> 30개 이하 평가한 유저 제거
rating_per_user_df = pd.DataFrame(ratings_per_anime)
filtered_rating_per_user_df = rating_per_user_df[rating_per_user_df.rating >= 30]
rating_users = filtered_rating_per_user_df.index.tolist()


# filtered_rating = pd.merge(rating, anime, on='anime_id')

# data filtering
filtered_rating = rating[rating.anime_id.isin(popular_anime)]
filtered_rating = rating[rating.user_id.isin(rating_users)]
filtered_rating = pd.merge(filtered_rating,  anime, on='anime_id')
# filtered_rating['anime_id'] = filtered_rating['name']
# filtered_rating = filtered_rating.drop('name', axis=1)

# 유저가 시청했지만 평가하지 않은 데이터 삭제
# 나중에 except로 유사도에서 제외시키기
df_hybrid = filtered_rating[filtered_rating != -1].dropna().reset_index().drop(['index'], axis=1)



# Split train- & testset
n = 100000
# df_hybrid = filtered_rating.sample(frac=1)
df_hybrid_train = df_hybrid.iloc[:100000, 0:3]
df_hybrid_test = df_hybrid.iloc[-n:, 0:3]

# filtered_rating pivot table 화
anime_matrix = filtered_rating.pivot_table(index='user_id', columns='anime_id', values='rating_x').fillna(0)
print('Shape User-Anime-Matrix:\t{}'.format(anime_matrix.shape))
anime_matrix.sample(3)

# Create user- & movie-id mapping
# user/movie_id (1, user_id) (1, movie_id) ~ enumerate mapping
user_anime_mapping = filtered_rating.copy()

user_id_mapping = {id:i for i, id in enumerate(user_anime_mapping['user_id'].unique())}
# id_user_mapping = {i: id for id, i in user_id_mapping.items()}
anime_id_mapping = {id:i for i, id in enumerate(user_anime_mapping['anime_id'].unique())}
# id_anime_mapping = {i: id for id, i in anime_id_mapping.items()}/


# Use mapping to get better ids
user_anime_mapping['user_id'] = user_anime_mapping['user_id'].map(user_id_mapping)
user_anime_mapping['anime_id'] = user_anime_mapping['anime_id'].map(anime_id_mapping)
user_anime_mapping

### 데이터 feature vector화
# feature_rateing = 
filtered_feature = anime.copy()
filtered_feature = filtered_feature[['anime_id','genre']].set_index('anime_id')

tfidf = TfidfVectorizer()
tfidf_hybrid = tfidf.fit_transform(filtered_feature['genre'].values.astype('U'))
print(tfidf_hybrid)

# movie_id와 user_id를 맵핑
mapping = {id:i for i, id in enumerate(filtered_feature.index)}

train_tfidf = []
# Iterate over all movie-ids and save the tfidf-vector
for id in df_hybrid_train['anime_id'].values:
    index = mapping[id]
    train_tfidf.append(tfidf_hybrid[index])
    
test_tfidf = []
# Iterate over all movie-ids and save the tfidf-vector
for id in df_hybrid_test['anime_id'].values:
    index = mapping[id]
    test_tfidf.append(tfidf_hybrid[index])
print(test_tfidf)

# stack
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
user_id_input = Input(shape=[1], name='user_id')
movie_id_input = Input(shape=[1], name='movie_id')
tfidf_input = Input(shape=[48], name='tfidf', sparse=True)

# Create separate embeddings for users and movies
user_embedding = Embedding(output_dim=user_embed,
                           input_dim=len(user_id_mapping),
                           input_length=1,
                           name='user_embedding')(user_id_input)
movie_embedding = Embedding(output_dim=movie_embed,
                            input_dim=len(anime_id_mapping),
                            input_length=1,
                            name='movie_embedding')(movie_id_input)

# Dimensionality reduction with Dense layers
# tfidf_vectors = Dense(2048, activation='relu')(tfidf_input)
# tfidf_vectors = Dense(512, activation='relu')(tfidf_input)
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
# csr_matrix to numpy array

# what is csr_matrix?? ㅠㅠㅠ
# steps = 10
# for i in range(steps):
#   # For simplicity, we directly use trainX and trainY in this example
#   # Usually, this is where batches are prepared
#   print(model.train_on_batch([df_hybrid_train['user_id'], df_hybrid_train['anime_id'], train_tfidf],
#                     df_hybrid_train['rating_x']))
model.fit([df_hybrid_train['user_id'], df_hybrid_train['anime_id'], train_tfidf],
                    df_hybrid_train['rating_x'],
                    batch_size=256, 
                    epochs=10,
                    validation_split=0.2,
                    shuffle=True,
                    verbose=1
                    )

model.save('/home/juno/workspace/user_collaborative_filtering/model_save/best_model_history/hybrid_deeplearning_movie3.h5')
np.save('/home/juno/workspace/user_collaborative_filtering/model_save/best_model_history/hybrid_deeplearning_movie3.npy', model.history)

y_pred = model.predict([df_hybrid_test['user_id'], df_hybrid_test['anime_id'], test_tfidf])
y_true = df_hybrid_test['Rating'].values

mse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False))
print('\n\nKeras Hybrid Deep Learning: {:.4f} RMSE'.format(mse))
print('\n\nKeras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))