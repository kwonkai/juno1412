
import pandas as pd

# To do linear algebra
import numpy as np

from tensorflow.keras.models import Model, load_model

import tensorflow as tf

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
from sklearn.preprocessing import LabelEncoder

# To use recommender systems
import surprise as sp
from surprise.model_selection import cross_validate

# To create deep learning models
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model, load_model

# To create sparse matrices
from scipy.sparse import coo_matrix

# To light fm
# from lightfm import LightFM
# from lightfm.evaluation import precision_at_k

# To stack sparse matrices
from scipy.sparse import vstack

# array 값 체인에 필요
import itertools

# 시드값 고정
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)


# df_hybrid.to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/df_hybrid.csv', index=False)
df_hybrid = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/df_hybrid.csv')
df_hybrid = pd.DataFrame(df_hybrid)
# df_hybrid['Rating'] = df_hybrid['Rating']/5/

# df_hybrid = df_hybrid.sample(frac=1)


## z-분포 scaler()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(df_hybrid[['Rating']])
scaler = scaler.transform(df_hybrid[['Rating']])
scaler = pd.DataFrame(list(itertools.chain(*scaler)), columns=['scaler'])

df_hybrid_z = pd.concat([ df_hybrid, scaler], axis=1).drop('Rating', axis=1)


## minmaxscaler()
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_scale_rating = minmax_scaler.fit_transform(df_hybrid[['Rating']])
minmax_scale_rating = pd.DataFrame(list(itertools.chain(*minmax_scale_rating )), columns=['scaler'])

df_hybrid_minmax = pd.concat([ df_hybrid, minmax_scale_rating], axis=1).drop('Rating', axis=1)


# train_test_split
n = 100000
# z분포
df_hybrid_train = df_hybrid_z[:100000]
df_hybrid_test = df_hybrid_z[-n:]

# minmax
df_hybrid_train = df_hybrid_minmax[:100000]
df_hybrid_test = df_hybrid_minmax[-n:]


# np.save('/home/juno/workspace/user_collaborative_filtering/data_files/train_tfidf_df.npy', train_tfidf_df)
# np.save('/home/juno/workspace/user_collaborative_filtering/data_files/test_tfidf_df.npy', test_tfidf_df)


train_tfidf_df = np.load('/home/juno/workspace/user_collaborative_filtering/data_files/train_tfidf_df.npy')
test_tfidf_df = np.load('/home/juno/workspace/user_collaborative_filtering/data_files/test_tfidf_df.npy')
train_tfidf_df = pd.DataFrame(train_tfidf_df)
test_tfidf_df = pd.DataFrame(test_tfidf_df)

# df_hybrid_train['User'] = df_hybrid_train['User'].values.astype(int).tolist()
# df_hybrid_train['Movie'] = df_hybrid_train['Movie'].values.astype(int).tolist()
# df_hybrid_test['User'] = df_hybrid_test['User'].values.astype(int).tolist()
# df_hybrid_test['Movie'] = df_hybrid_test['Movie'].values.astype(int).tolist()

# train_tfidf_df.to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/train_tfidf_df.csv')
# train_tfidf_df = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/train_tfidf_df.csv')

# test_tfidf_df = pd.DataFrame(test_tfidf)/
# test_tfidf_df.to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/test_tfidf_df.csv')
# test_tfidf_df = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/test_tfidf_df.csv')




# 모델 복원
loaded_model = tf.keras.models.load_model('/home/juno/workspace/user_collaborative_filtering/model_save/best_model_history/kaggle2_normalization_epoch3_hybrid_deeplearning_movie.h5')
loaded_model.summary()

y_pred = loaded_model.predict([df_hybrid_test['User'], df_hybrid_test['Movie'], test_tfidf_df])
print('-------------df_hybrid_test--------------------')
print(df_hybrid_test)
print('-------------df_hybrid_user--------------------')
print(df_hybrid_test['User'])
print('-------------df_hybrid_movie--------------------')
print(df_hybrid_test['Movie'])
print('-------------test_tfidf--------------------')
print(test_tfidf_df)
print('-------------y_pred--------------------')
print(y_pred)




y_preds = y_pred.tolist()
y_preds_list = list(itertools.chain(*y_preds))
y_preds_df = pd.DataFrame(y_preds_list).reset_index().drop('index', axis=1)
# y_preds_df.to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/recommend_data/y_preds_zscore_df.csv')


# df_hybrid_test.reset_index().drop('index', axis=1).to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/recommend_data/df_hybrid_test_zscore.csv')


y_preds_test_merge = pd.concat([y_preds_df, df_hybrid_test])
# y_preds_test_merge.to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/recommend_data/y_preds_test_merge.csv')


y_true = df_hybrid_test['scaler'].values

mse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False))
print('\n\nKeras Hybrid Deep Learning: {:.4f} MSE'.format(mse))
print('\n\nKeras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))






# train_test_split
df_hybrid_train = df_hybrid_minmax[:100000]
df_hybrid_test = df_hybrid_minmax[-n:]


train_tfidf_df = np.load('/home/juno/workspace/user_collaborative_filtering/data_files/train_tfidf_df.npy')
test_tfidf_df = np.load('/home/juno/workspace/user_collaborative_filtering/data_files/test_tfidf_df.npy')
train_tfidf_df = pd.DataFrame(train_tfidf_df)
test_tfidf_df = pd.DataFrame(test_tfidf_df)





# 모델 복원
loaded_model = tf.keras.models.load_model('/home/juno/workspace/user_collaborative_filtering/model_save/best_model_history/kaggle2_normalization_epoch3_hybrid_deeplearning_movie.h5')
loaded_model.summary()

y_pred = loaded_model.predict([df_hybrid_test['User'], df_hybrid_test['Movie'], test_tfidf_df])
print('-------------df_hybrid_test--------------------')
print(df_hybrid_test)
print('-------------df_hybrid_user--------------------')
print(df_hybrid_test['User'])
print('-------------df_hybrid_movie--------------------')
print(df_hybrid_test['Movie'])
print('-------------test_tfidf--------------------')
print(test_tfidf_df)
print('-------------y_pred--------------------')
print(y_pred)




y_preds = y_pred.tolist()
y_preds_list = list(itertools.chain(*y_preds))
y_preds_df = pd.DataFrame(y_preds_list).reset_index().drop('index', axis=1)
y_preds_df.to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/recommend_data/y_preds_minmax_df.csv')


df_hybrid_test.reset_index().drop('index', axis=1).to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/recommend_data/df_hybrid_test_minmax.csv')


y_preds_test_merge = pd.concat([y_preds_df, df_hybrid_test])
# y_preds_test_merge.to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/recommend_data/y_preds_test_merge.csv')


y_true = df_hybrid_test['scaler'].values

mse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False))
print('\n\nKeras Hybrid Deep Learning: {:.4f} MSE'.format(mse))
print('\n\nKeras Hybrid Deep Learning: {:.4f} RMSE'.format(rmse))
