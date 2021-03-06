# To store the data
from pickletools import float8
import pandas as pd

# To do linear algebra
import numpy as np


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



# Load data for all movies
movie_titles = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/netflix/movie_titles.csv', 
                           encoding = 'ISO-8859-1', 
                           header = None, 
                           names = ['Id', 'Year', 'Name']).set_index('Id')

print('Shape Movie-Titles:\t{}'.format(movie_titles.shape))


# Load a movie metadata dataset
movie_metadata = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/movies_metadata.csv', low_memory=False)[['original_title', 'overview', 'vote_count']].set_index('original_title').dropna()
# Remove the long tail of rarly rated moves
movie_metadata = movie_metadata[movie_metadata['vote_count']>10].drop('vote_count', axis=1)
print('Shape Movie-Metadata:\t{}'.format(movie_metadata.shape))



# Load single data-file
df_raw = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/netflix/combined_data_1.txt', header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])
# Find empty rows to slice dataframe for each movie
# df_raw['Rating'] ??? ?????????(1:, 2???, 3:) ??????
tmp_movies = df_raw[df_raw['Rating'].isna()]['User'].reset_index()
movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]

# Shift the movie_indices by one to get start and endpoints of all movies
# deque : ????????? ?????????
# rotate(-1) : ????????? ?????? ?????? [1, 2, 3] -> -1 : [2, 3, 1], 1: [3, 1, 2]
shifted_movie_indices = deque(movie_indices)
shifted_movie_indices.rotate(-1)



# Gather all dataframes
user_data = []

# Iterate over all movies
# movie_indice = ??????, shift_movie_indices = ??? 2??? ?????? ????????? movie_indence
for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):
    
    # Check if it is the last movie in the file
    # ?????? ????????? ????????? 1?????? retato??? ????????? ?????? ????????? ?????? ???????????? movie_id ???????????? ????????????.
    if df_id_1<df_id_2:
        tmp_df = df_raw.loc[df_id_1+1:df_id_2-1].copy()
    else:
        tmp_df = df_raw.loc[df_id_1+1:].copy()
    
    # movie_id ?????? ?????????
    tmp_df['Movie'] = movie_id
    
    # ??? user_data ???????????? tmp_df[user, rating, date, movie_id] ?????? ?????????
    user_data.append(tmp_df)

# Combine all dataframes
df = pd.concat(user_data)

'''
# like/unlike ?????? 1 (0 : 1???, 2??? / 1 : 3???, 4???, 5???)
# rating ????????? type ?????? ???, replace??? rating ?????? ????????????
df['Rating'] = df['Rating'].astype(int)
df = df.replace({'Rating':{1:0, 2:0, 3:1, 4:1, 5:1}})
del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id
print('Shape User-Ratings:\t{}'.format(df.shape))
df.sample(5)
'''
# like/unlike ?????? 2
# Rating : 1(???????????? ??????) | Rating 2,3(?????????) | Rating 4,5(?????????)
# Rating : -1 or NaN(???????????? ??????) | Rating 0(?????????) | Rating 1(?????????)
df['Rating'] = df['Rating'].astype(int)
# df = df.replace({'Rating':{1:0, 2:0, 3:0, 4:1, 5:1}})
del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id
print('Shape User-Ratings:\t{}'.format(df.shape))
df.sample(5)


# Filter sparse movies
min_movie_ratings = 10000
filter_movies = (df['Movie'].value_counts()>min_movie_ratings)
filter_movies = filter_movies[filter_movies].index.tolist()

# Filter sparse users
min_user_ratings = 200
filter_users = (df['User'].value_counts()>min_user_ratings)
filter_users = filter_users[filter_users].index.tolist()

# Actual filtering
df_filterd = df[(df['Movie'].isin(filter_movies)) & (df['User'].isin(filter_users))]
del filter_movies, filter_users, min_movie_ratings, min_user_ratings
print('Shape User-Ratings unfiltered:\t{}'.format(df.shape))
print('Shape User-Ratings filtered:\t{}'.format(df_filterd.shape))



# # Shuffle DataFrame
# df_filterd = df_filterd.drop('Date', axis=1).sample(frac=1).reset_index(drop=True)
# # Testingsize
# n = 100000
# # Split train- & testset
# df_train = df_filterd[:-n]
# df_test = df_filterd[-n:]



# Create a user-movie matrix with empty values
# df_p = df_train.pivot_table(index='User', columns='Movie', values='Rating')
# print('Shape User-Movie-Matrix:\t{}'.format(df_p.shape))
# df_p.sample(3)

# Create user- & movie-id mapping
user_id_mapping = {id:i for i, id in enumerate(df['User'].unique())}
movie_id_mapping = {id:i for i, id in enumerate(df['Movie'].unique())}

# Use mapping to get better ids
df['User'] = df['User'].map(user_id_mapping)
df['Movie'] = df['Movie'].map(movie_id_mapping)


##### Combine both datasets to get movies with metadata
# Preprocess metadata
# metadata : title, overview ??????
tmp_metadata = movie_metadata.copy()
tmp_metadata.index = tmp_metadata.index.str.lower()
print(tmp_metadata)

# Preprocess titles
tmp_titles = movie_titles.drop('Year', axis=1).copy()
# ?????? index = id -> index reset ??? name(????????????) ????????? ???????????? ??????
# name(????????????) ????????? ???????????? ??????
tmp_titles = tmp_titles.reset_index().set_index('Name')
tmp_titles.index = tmp_titles.index.str.lower()
print(tmp_titles)

# Combine titles and metadata
# ?????? ?????? ???????????? metadata(?????? ??????)??? join
# dropna()??? ????????? ?????? ???, index = Id??? ??????
# overview ????????? string??? ???????????? ?????????
# df_id_descriptions = join?????? ????????? ????????? movid_id : 17770 -> 6938??? ??????
df_id_descriptions = tmp_titles.join(tmp_metadata).dropna().set_index('Id')
df_id_descriptions['overview'] = df_id_descriptions['overview'].str.lower()
print(df_id_descriptions)
del tmp_metadata,tmp_titles


# Filter all ratings with metadata
# df_id_descriptions : ?????? ??????
# df?????? data ?????? ?????? ??? movie ????????? index??? ?????? -> 4499?????? ????????? ?????? user ?????? ???????????? ???????????? ????????? ??????
# df_id_description??? 'Id'(=movie_id)??? ???????????? overview ??????
# ???????????? ??????????????? index ????????? Movie??? ?????? ????????????
# df_hybrid = Movie User Rating??? ?????? ?????? 880????????? ???????????? ????????? ??? 
df_hybrid1 = df.drop('Date', axis=1).set_index('Movie').join(df_id_descriptions).dropna().drop('overview', axis=1)
df_hybrid = df_hybrid1.reset_index().rename({'index':'Movie'}, axis=1)
print(df)
print(df_id_descriptions)
print(df_hybrid1)
print(df_hybrid)

# data normalization(????????? ????????? : 0.2, 0.4, 0.6, 0.8, 1.0)
df_hybrid['Rating'] = df_hybrid['Rating'] / 5

# Split train- & testset
n = 100000
df_hybrid = df_hybrid.sample(frac=1).reset_index(drop=True)
df_hybrid_train = df_hybrid[:100000]
df_hybrid_test = df_hybrid[-n:]


# tf_idf (??????-?????? ?????????) ??????
# STOP_WORD =? 
tfidf = TfidfVectorizer(stop_words='english')
print(tfidf)
tfidf_hybrid = tfidf.fit_transform(df_id_descriptions['overview'])
print(tfidf_hybrid)

# Get mapping from movie-ids to indices in tfidf-matrix
# idx??? movie_id??? mapping
# ????????? ?????? + ????????? ??????
# ????????? 685, ????????? 6254
# df_id_descriptions1 = df_id_descriptions.duplicated(keep='first').value_counts()/
# df_id_descriptions1 = pd.concat([df_id_descriptions, df_id_descriptions1], axis=1)


# mapping = movie_id, idx mapping, 
# ????????? ?????? 6938??? -> 6104 : unique() ?????? ?????? ?????????
# id ?????? + overview ?????? ??????
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
# sparse ??????
# 1: [0,...,25] 2:[0,...35] -> 
train_tfidf = vstack(train_tfidf)
print(train_tfidf)
test_tfidf = vstack(test_tfidf)
train_tfidf = train_tfidf.toarray()
test_tfidf = test_tfidf.toarray()



#### dataprame to list
# df_hybrid['User'] = df_hybrid['User'].astype(int).tolist()
# df_hybrid['Movie'] = df_hybrid['Movie'].astype(int).tolist()




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
          epochs=3,
          validation_split=0.2,
          shuffle=True,
          verbose=1)


model.save('/home/juno/workspace/user_collaborative_filtering/model_save/best_model_history/re_normalizationg_epoch3_hybrid_deeplearning_movie3.h5')
# np.save('/home/juno/workspace/user_collaborative_filtering/model_save/best_model_history/hybrid_deeplearning_movie3.npy', model.history)


## ????????? ???   
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