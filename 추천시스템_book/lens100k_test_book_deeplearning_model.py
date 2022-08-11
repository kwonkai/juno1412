# Hybrid 추천 - CF + MF

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# u.item 파일을 DataFrame으로 읽기
movies = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_reco_book/movielens_100k_csv/movies.csv')

# u.data 파일을 DataFrame으로 읽기
ratings = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_reco_book/movielens_100k_csv/ratings.csv') 

# train test 분리
# train test 분리
from sklearn.utils import shuffle
TRAIN_SIZE = 0.8
ratings = shuffle(ratings)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

# u.user 파일을 DataFrame으로 읽기 
# 수정된 부분 1 >>>>>>>>>>
u_cols = ['userId', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_reco_book/u.user', sep='|', names=u_cols, encoding='latin-1')
users = users[['userId', 'occupation']]

# Convert occupation(string to integer)
# 직업명 labeling작업 ()
occupation = {}
def convert_occ(x):
    if x in occupation:
        return occupation[x]
    else:
        occupation[x] = len(occupation)
        return occupation[x]
users['occupation'] = users['occupation'].apply(convert_occ)

L = len(occupation)
train_occ = pd.merge(ratings_train, users, on='userId')['occupation']
# train_occ = train_occ.drop(0, axis=0)
test_occ = pd.merge(ratings_test, users, on='userId')['occupation']



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
K = 200                             # Latent factor 수 
mu = ratings_train.rating.mean()    # 전체 평균 
# M = 34000 + 1       # Number of users
# N = 34910 + 1      # Number of movies

M = ratings.userId.max() + 1       # Number of users
N = ratings.movieId.max() + 1      # Number of movies

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
occ = Input(shape=(1, ))
occ_embedding = Embedding(L, 3, embeddings_regularizer=l2())(occ)
occ_layer = Flatten()(occ_embedding)
R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias, occ_layer])
#<<<<<<<<< 수정된 부분 2

# Neural network
R = Dense(2048)(R)
R = Activation('relu')(R)
R = Dense(256)(R)
R = Activation('linear')(R)
R = Dense(1)(R)

# 수정된 부분 3 >>>>>>>>>>
model = Model(inputs=[user, item, occ], outputs=R)
#<<<<<<<<< 수정된 부분 3
model.compile(
  loss=RMSE,
  optimizer=SGD(),
  #optimizer=Adamax(),
  metrics=[RMSE]
)
model.summary()

# Model fitting
result = model.fit(
  x=[ratings_train.userId.values, ratings_train.movieId.values, train_occ.values],
  y=ratings_train.rating.values - mu,
  epochs=80,
  batch_size=512,
  validation_data=(
    [ratings_test.userId.values, ratings_test.movieId.values, test_occ.values],
    ratings_test.rating.values - mu
  )
)

# Plot RMSE
import matplotlib.pyplot as plt
plt.plot(result.history['RMSE'])
plt.plot(result.history['val_RMSE'])
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('movielens_100k_traintes_rmse2.png')



# Prediction
user_ids = ratings_test.userId.values[0:6]
movie_ids = ratings_test.movieId.values[0:6]
user_occ = test_occ[0:6]
predictions = model.predict([user_ids, movie_ids, user_occ]) + mu
print("Actuals: \n", ratings_test[0:6])
print()
print("Predictions: \n", predictions)

# 정확도(RMSE)를 계산하는 함수 
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

user_ids = ratings_test.userId.values
movie_ids = ratings_test.movieId.values
y_pred = model.predict([user_ids, movie_ids, test_occ]) + mu
y_pred = np.ravel(y_pred, order='C')
y_true = np.array(ratings_test.rating)

RMSE2(y_true, y_pred)


test = y_pred.reshape(20001,)
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

plt.savefig('movielens_100k_deeplearning_test3.png')

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