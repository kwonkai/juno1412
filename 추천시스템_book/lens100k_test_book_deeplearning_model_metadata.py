# Hybrid 추천 - CF + MF

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Split train- & testset
# 시드값 고정
import tensorflow as tf
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# hybrid_df 데이터 가져오기
hybrid_df = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/metadata_shuffle_feature1.csv')
hybrid_df['rating'] = hybrid_df['rating'].round().replace([0],1)
hybrid_df = hybrid_df[:500000]

## 장르 벡터화
########### 장르 전처리 #######
# 장르 -> [0,1,0,1,0,0,1,.....]
## 중첩리스트 제거 & 방송국 지우기
from ast import literal_eval 
hybrid_df['genres'] = hybrid_df['genres'].apply(literal_eval)
hybrid_df['genre_name'] = hybrid_df['genres'].apply(lambda x : [ y['name'] for y in x])


# df_id_overview = hybrid_df_50[['id', 'overview']].set_index('id')


genre_feature = pd.DataFrame(hybrid_df['genre_name'].to_list())
# genre_feature.replace({'Animation':1},{'Adventure':2},{'Romance':3},{'Comedy':4},{'Family':5},{'History':6})
# genre_feature.replace({'Crime':8},{'Fantasy':9},{'Science Fiction':10},{'Thriller':11},{'Music':12},{'Horror':13},{'Documentary':14})
# genre_feature.replace({'Mystery':15},{'Western':16},{'TV Movie':17},{'War':18},{'Foreign':19},{'Drama':7})

genre_feature.columns = ['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7']



remove_list = genre_feature.isin(['Carousel Productions', 'Vision View Entertainment', 'Telescene Film Group Productions',\
                                   'Aniplex', 'GoHands', 'BROSTA TV', 'Mardock Scramble Production Committee', 'Sentai Filmworks', \
                                   'Odyssey Media', 'Pulser Productions', 'Rogue State', 'The Cartel'])
               

# genre 유일값 확인하기 : 방송국 제거되었는지 확인
print(pd.unique(genre_feature[['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7']].values.ravel()))


genre_feature['genre_feature'] = genre_feature.fillna('').apply(lambda x:','.join(x), axis=1)


hybrid_df = hybrid_df.reset_index().join(genre_feature['genre_feature']).set_index('index').drop(columns=['overview','genre_name', 'genres'])
movies = hybrid_df[['userId', 'id', 'rating', 'title', 'tagline', 'genre_feature']]
movies.columns = ['userId', 'movieId', 'rating', 'title', 'tagline', 'genre_feature']


# u.user 파일을 DataFrame으로 읽기 
# 수정된 부분 1 >>>>>>>>>>

# Convert occupation(string to integer)
# 장르 labeling작업 ()
genre = {}
def convert_genre(x):
    if x in genre:
        return genre[x]
    else:
        genre[x] = len(genre)
        return genre[x]
movies['genre_feature'] = movies['genre_feature'].apply(convert_genre)

L1 = len(genre)


# 제목 labeling작업 ()
title = {}
def convert_title(x):
    if x in title:
        return title[x]
    else:
        title[x] = len(title)
        return title[x]
movies['title'] = movies['title'].apply(convert_title)

L2 = len(title)


# 태그 labeling작업 ()
tag = {}
def convert_tag(x):
    if x in tag:
        return tag[x]
    else:
        tag[x] = len(tag)
        return tag[x]
movies['tagline'] = movies['tagline'].apply(convert_tag)

L3 = len(tag)

# train test 분리
# train test 분리
from sklearn.utils import shuffle
TRAIN_SIZE = 0.80
cutoff = int(TRAIN_SIZE * len(hybrid_df))
ratings_train = movies.iloc[:cutoff]
ratings_test = movies.iloc[cutoff:]

train_genre = ratings_train['genre_feature']
test_genre = ratings_test['genre_feature']
train_title = ratings_train['title']
test_title = ratings_test['title']
train_tag = ratings_train['tagline']
test_tag = ratings_test['tagline']
# metadata = pd.merge(users, ratings, on='userId')
# metadata_df = pd.merge(metadata, movies, on='movieId')


# # mapping 중요!! 
# # 무조건 있어야 할 것!
# # Create user- & movie-id mapping
# user_id_mapping = {id:i for i, id in enumerate(metadata_df['userId'].unique())}
# movie_id_mapping = {id:i for i, id in enumerate(metadata_df['movieId'].unique())}

# # Use mapping to get better ids
# metadata_df['userId'] = metadata_df['userId'].map(user_id_mapping)
# metadata_df['movieId'] = metadata_df['movieId'].map(movie_id_mapping)

# 추가수정 # ratings_train 갯수 맞추기 # user_id에서 user 데이터 차이로 occ 데이터 개수가 맞지 않음
# ratings_train = ratings_train[:26162]
#<<<<<<<<< 수정된 부분 1

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam, Adamax

# Variable 초기화 
K = 100                            # Latent factor 수 
mu = ratings_train.rating.mean()    # 전체 평균 
# M = 34000 + 1       # Number of users
# N = 34910 + 1      # Number of movies

M = movies.userId.max() + 1       # Number of users
N = movies.movieId.max() + 1      # Number of movies

# Defining RMSE measure
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

##### (2)

# Keras model
user = Input(shape=(1, ))
item = Input(shape=(1, ))
P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user)
Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item)
user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user)
item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item)

# Concatenate layers
from tensorflow.keras.layers import Dense, Concatenate, Activation
P_embedding = Flatten()(P_embedding)
Q_embedding = Flatten()(Q_embedding)
user_bias = Flatten()(user_bias)
item_bias = Flatten()(item_bias)

# 수정된 부분 2 >>>>>>>>>>
genre = Input(shape=(1, ))
genre_embedding = Embedding(L1, 3, embeddings_regularizer=l2())(genre)
genre_layer = Flatten()(genre_embedding)

title = Input(shape=(1, ))
title_embedding = Embedding(L2, 3, embeddings_regularizer=l2())(title)
title_layer = Flatten()(title_embedding)

tag = Input(shape=(1, ))
tag_embedding = Embedding(L3, 3, embeddings_regularizer=l2())(tag)
tag_layer = Flatten()(tag_embedding)
R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias, genre_layer, title_layer, tag_layer])
#<<<<<<<<< 수정된 부분 2

# Neural network
R = Dense(2048)(R)
R = Activation('relu')(R)
R = Dense(256)(R)
R = Activation('linear')(R)
R = Dense(1)(R)

# 수정된 부분 3 >>>>>>>>>>
model = Model(inputs=[user, item, genre, title, tag], outputs=R)
#<<<<<<<<< 수정된 부분 3
model.compile(
  loss=RMSE,
#   optimizer=SGD(),
  optimizer='adam',
  metrics=[RMSE]
)
model.summary()

# Model fitting
result = model.fit(
  x=[ratings_train.userId.values, ratings_train.movieId.values, train_genre.values, train_title.values, train_tag.values],
  y=ratings_train.rating.values ,
  epochs=2,
  batch_size=512,
  validation_split=0.2
)

# Plot RMSE
import matplotlib.pyplot as plt
plt.plot(result.history['RMSE'])
plt.plot(result.history['val_RMSE'])
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('new_metadata_genre_title_rmse5.png')



# Prediction
# user_ids = ratings_test.userId.values[0:6]
# movie_ids = ratings_test.movieId.values[0:6]
# user_occ = test_occ[0:6]
# predictions = model.predict([user_ids, movie_ids, user_occ]) + mu
# print("Actuals: \n", ratings_test[0:6])
# print()
# print("Predictions: \n", predictions)

# 정확도(RMSE)를 계산하는 함수 
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

user_ids = ratings_test.userId.values
movie_ids = ratings_test.movieId.values
y_pred = model.predict([user_ids, movie_ids, test_genre, test_title, test_tag])
y_pred = np.ravel(y_pred, order='C')
y_true = np.array(ratings_test.rating)

RMSE2(y_true, y_pred)


test = y_pred.reshape(100000,)
tmp = np.stack((y_true, test), axis=1)
print(tmp)
df_v = pd.DataFrame(tmp, columns=['true', 'pred'])
df_v.head(10)

result = df_v.sort_values(by=['true', 'pred'])
result =result.reset_index(drop=True)

# 그래프 그리기
y_pred = result['pred']
for i in range(len(y_pred)):
    if result['true'][i] > result['pred'][i]:
        plt.vlines(i, result['pred'][i], result['true'][i], color='red', linestyle='solid', linewidth=1)
    else:
        plt.vlines(i, result['true'][i], result['pred'][i], color='red', linestyle='solid', linewidth=1)
plt.xlabel('rating_count')
plt.ylabel('rating')

plt.savefig('new_metadata_all_bookmodel_deeplearning_test5_1_5_adam2.png')

# 그래프 크기 계산
y_pred = result['pred']
y_area = []
y_area1 = []
y_area2 = []
for i in range(len(y_pred)):
    if result['true'][i] > result['pred'][i]:
        y_area1_list = result['true'][i] - result['pred'][i]
        y_area1.append(y_area1_list)
    else:
        y_area2_list = result['pred'][i] - result['true'][i]
        y_area2.append(y_area2_list)
    y_area = y_area1 + y_area2

print(sum(y_area))

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
mse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False))
# rmsle = np.sqrt(mean_squared_log_error(y_pred=y_pred, y_true=y_true))/
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print('\n\nKeras Hybrid Deep Learning: {:.4f} MSE'.format(mse))
print('\n\nKeras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))
# print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} RMSLE'.format(rmsle))
print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} MAE'.format(mae))
print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} R2'.format(r2))









print('\n\nKeras Hybrid Deep Learning: {:.4f} MSE'.format(mse))
print('\n\nKeras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))
#######################################################################
# y_pred = model.predict([df_hybrid_test['userId'], df_hybrid_test['movieId'], test_tfidf_title.toarray()])
# y_true = df_hybrid_test['rating'].values
# y_pred = np.where(y_pred < 0.5, 0.5, np.where(y_pred > 5, 5, y_pred))


# test = y_pred.reshape(20000,)
# tmp = np.stack((y_true, test), axis=1)
# print(tmp)
# df_v = pd.DataFrame(tmp, columns=['true', 'pred'])
# df_v.head(10)

# result = df_v.sort_values(by=['true', 'pred'])
# result =result.reset_index(drop=True)
# result.to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/recommend_data/origin_overview_genre_test.csv')


# 그래프 그리기
# y_pred = result['pred']
# for i in range(len(y_pred)):
#     if result['true'][i] > result['pred'][i]:
#         plt.vlines(i, result['pred'][i], result['true'][i], color='red', linestyle='solid', linewidth=1)
#     else:
#         plt.vlines(i, result['true'][i], result['pred'][i], color='red', linestyle='solid', linewidth=1)
# plt.xlabel('rating_count')
# plt.ylabel('rating')

# plt.savefig('movielens_100k_originmodel_test.png')

# # 그래프 그리기
# y_pred = result['pred']
# y_area = []
# y_area1 = []
# y_area2 = []
# for i in range(len(y_pred)):
#     if result['true'][i] > result['pred'][i]:
#         y_area1_list = result['true'][i] - result['pred'][i]
#         y_area1.append(y_area1_list)
#     else:
#         y_area2_list = result['pred'][i] - result['true'][i]
#         y_area2.append(y_area2_list)
#     y_area = y_area1 + y_area2

# print(sum(y_area))

# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_log_error
# from sklearn.metrics import r2_score

# mse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
# rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False))
# # rmsle = np.sqrt(mean_squared_log_error(y_pred=y_pred, y_true=y_true))/
# mae = mean_absolute_error(y_true, y_pred)
# r2 = r2_score(y_true, y_pred)

# print('\n\nKeras Hybrid Deep Learning: {:.4f} MSE'.format(mse))
# print('\n\nKeras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))
# # print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} RMSLE'.format(rmsle))
# print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} MAE'.format(mae))
# print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} R2'.format(r2))


# rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
# print('\n\nTesting Result With Keras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))