#f라이브러리 설정
# To store the data
import enum
from pickletools import float8
from matplotlib.transforms import Transform
import pandas as pd

# To do linear algebra
import numpy as np


# To create plots
import matplotlib.pyplot as plt

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

# To stack sparse matrices
from scipy.sparse import vstack

# Split train- & testset
# 시드값 고정
import tensorflow as tf
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

########### 데이터 불러오기 & 잘못들어가 있는 값 제거
# id 대신 date값이 들어가있는 index 제거
movie_metadata = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/netflix/movies_metadata.csv', low_memory=False)[['id', 'original_title','overview']].drop([19730, 29503, 35587], axis=0).dropna()

# id 컬럼 type 변경
movie_metadata['id'] = movie_metadata['id'].astype(float).astype(int)

## rating 데이터 불러오기 & id기준으로 병합하기
rating = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/netflix/ratings.csv').rename(columns={"movieId" : "id"}).drop('timestamp', axis=1)
hybrid_df = pd.merge(rating, movie_metadata, on='id')

# id -> Movieid로 변경, rating = int로 변경(0~5 10단위 -> 5단위로 변경)
hybrid_df.rename(columns={'id' : 'Movieid'}, inplace=True)
hybrid_df = hybrid_df.sample(frac=1)[:1000]


# np.save('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/genre_hybrid_df.npy', hybrid_df)

## 데이터 정규화
## minmaxscaler()
import itertools
from sklearn.preprocessing import MinMaxScaler
# minmax_scaler = MinMaxScaler()
# minmax_scale_rating = minmax_scaler.fit_transform(hybrid_df[['rating']])
# minmax_scale_rating = pd.DataFrame(list(itertools.chain(*minmax_scale_rating )), columns=['scaler'])

# df_hybrid_minmax = pd.concat([hybrid_df, minmax_scale_rating], axis=1).drop('rating', axis=1)


# moivie_id, overview 고유값만 있는 dataframe 생성
df_hybrid_minmax = hybrid_df[['userId', 'Movieid', 'rating', 'overview']]
df_hybrid_train = df_hybrid_minmax[:1000] 
df_hybrid_test = df_hybrid_minmax[-100:]


df_id_overview = df_hybrid_minmax[['Movieid', 'overview']].set_index('Movieid')
# df_id_overview = df_hybrid_minmax[['Movieid', 'original_title']].set_index('Movieid')
# df_id_overview = df_hybrid_minmax[['Movieid', 'overview']].drop_duplicates().set_index('Movieid')
# df_id_overview_train = df_hybrid_train[['Movieid', 'overview']].set_index('Movieid')
# df_id_overview_test = df_hybrid_test[['Movieid', 'overview']].set_index('Movieid')



## 단어 벡터화
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('popular')
# overview의 모든 문장-단어를 tokenizer : 단어마다 분리하여 리스트로 분리
df_id_overview['token'] = df_id_overview.apply(lambda row: word_tokenize(row['overview']), axis=1)

# token 컬럼을 list화 시켜 데이터 프레임으로 저장
# token의 모든 단어가 dataframe의 value 값으로 들어감
title_token_df = pd.DataFrame(df_id_overview['token'].to_list())

# 193개의 컬럼의 모든 value 값을 반복하여 overtoken 리스트에 모든 단어 값을 추가해줌
all_word = []
for i in range(len(title_token_df.columns)):
    token = title_token_df[i].unique().tolist()
    for j in token:
        all_word.append(j)

# 각 column 별로 list화 시킨 list 만들기
overview_list_token = []
for i in range(len(title_token_df.index)):
    token = title_token_df.loc[i].unique().tolist()
    overview_list_token.append(token)

# 각 단어별 mapping
mapping = {id: i for i, id in enumerate(all_word)}
# word_to_index = {all_word[0] : index for index, all_word in enumerate(all_word)}

# 각 list별로 정수 맵핑하기
encoded_token = []
for overview in overview_list_token:
    tmp1 = []
    tmp2 = []
    for word in overview:
        try:
            # 각 글자를 해당하는 정수로 변환한다.
            tmp1.append(mapping[word])
            tmp2.append(tmp1)
        except KeyError: 
            pass

    encoded_token.append(tmp2)


encoded_token_df = pd.DataFrame(encoded_token)[0]
df = []
for i in 1000:
    df.append(list(map(int, encoded_token_df.loc[i])))
# token_df = encoded_token_df.apply(lambda x: ','.join(x), axis=1)
encoded_token_df.columns = ['overview']

df_id_overview = pd.concat([df_hybrid_minmax.reset_index(), encoded_token_df], axis=1).drop(columns=['overview'], axis=1).rename(columns={ 0 :'overview'})


# mapping = {i: id for i, id in enumerate(overview_token)}
# over_view_label = []
# for i in mapping():
#     over_view_label.append(mapping[i])


# overview_token_df = pd.DataFrame(overview_token, columns=[''])

# title_token_df1 = title_token_df[0].unique().tolist() + title_token_df[1].unique().tolist() + title_token_df[2].unique().tolist() + title_token_df[3].unique().tolist() + title_token_df[4].unique().tolist() + \
                #   title_token_df[5].unique().tolist() + title_token_df[6].unique().tolist() + title_token_df[7].unique().tolist() + title_token_df[8].unique().tolist() + title_token_df[9].unique().tolist()
 


# tfidf 문장-단어 벡터화 : 문장에 있는 모든 단어 벡터화
# tfidf = TfidfVectorizer()
# tfidf_title = tfidf.fit_transform(df_id_overview['overview'])


# tfidf_overview = tfidf.fit_transform(df_hybrid_train['original_title'])
# tfidf_overview_df = pd.DataFrame(tfidf_overview.toarray())
# tfidf_overview_df = tfidf_overview_df.astype(str)
# tfidf_overview_df['feature'] = tfidf_overview_df.fillna('').apply(lambda x: ','.join(x), axis=1)



# tfidf_overview_df = pd.DataFrame(tfidf_overview.toarray())
# tfidf_overview_df = tfidf_overview_df.astype(str)
# tfidf_overview_df['feature'] = tfidf_overview_df.fillna('').apply(lambda x: ','.join(x), axis=1)
# df_hybrid_train = df_hybrid_train.reset_index().drop(columns=['index'], axis=1)
# df_hybrid_train = pd.concat([df_hybrid_train, tfidf_overview_df['feature']],axis=1)

# train_tfidf = vstack(tfidf_overview)
# tf.sparse.reorder(train_tfidf)
# train_tfidf = train_tfidf.todense()
'''
# 오버뷰 tfidf
tfidf = TfidfVectorizer()
# tfidf_genre1 = tfidf.fit_transform(df_hybrid_minmax['overview'])
# tfidf_genre2 = tfidf.fit_transform(df_hybrid_test['overview'])
tfidf_genre = tfidf.fit_transform(df_id_overview['overview'])

# mapping1 = {id:i for i, id in enumerate(df_hybrid_train['Movieid'])}
# mapping2 = {id:i for i, id in enumerate(df_hybrid_test['Movieid'])}
mapping = {id:i for i, id in enumerate(df_id_overview['overview'].index)}

train_tfidf = []
# Iterate over all movie-ids and save the tfidf-vector
for id in df_hybrid_train['Movieid'].values:
    index = mapping[id]
    train_tfidf.append(tfidf_genre[index])
    
test_tfidf = []
# Iterate over all movie-ids and save the tfidf-vector
for id in df_hybrid_test['Movieid'].values:
    index = mapping[id]
    test_tfidf.append(tfidf_genre[index])


# Stack the sparse matrices
train_tfidf = vstack(train_tfidf)
test_tfidf = vstack(test_tfidf)
train_tfidf = train_tfidf.toarray()
test_tfidf = test_tfidf.toarray()




# Create user- & movie-id mapping
user_id_mapping = {id:i for i, id in enumerate(df_hybrid_minmax['userId'].unique())}
movie_id_mapping = {id:i for i, id in enumerate(df_hybrid_minmax['Movieid'].unique())}

# Use mapping to get better ids
df_hybrid_minmax['userId'] = df_hybrid_minmax['userId'].map(user_id_mapping)
df_hybrid_minmax['Movieid'] = df_hybrid_minmax['Movieid'].map(movie_id_mapping)

# Use mapping to get better ids
# df_hybrid_z['userId'] = df_hybrid_z['userId'].map(user_id_mapping)
# df_hybrid_z['Movie'] = df_hybrid_z['Movie'].map(movie_id_mapping)
'''

##### Setup the network
# Network variables
user_embed = 10
movie_embed = 10


# Create two input layers
user_id_input = Input(shape=[1], name='user')
movie_id_input = Input(shape=[1], name='movie')
overview_input = Input(shape=[1], name = 'overview')

# Create separate embeddings for users and movies
user_embedding = Embedding(output_dim=user_embed,
                           input_dim=1000,
                           input_length=1,
                           name='user_embedding')(user_id_input)
movie_embedding = Embedding(output_dim=movie_embed,
                            input_dim=1000,
                            input_length=1,
                            name='movie_embedding')(movie_id_input)

# Dimensionality reduction with Dense layers
overview_vectors = Dense(128, activation='relu')(overview_input)
overview_vectors = Dense(32, activation='relu')(overview_vectors)


# Reshape both embedding layers
user_vectors = Reshape([user_embed])(user_embedding)
movie_vectors = Reshape([movie_embed])(movie_embedding)

# Concatenate all layers into one vector
both = Concatenate()([user_vectors, movie_vectors, overview_vectors])

# Add dense layers for combinations and scalar output
dense = Dense(512, activation='relu')(both)
dense = Dropout(0.2)(dense)
output = Dense(1)(dense)


# Create and compile model
model = Model(inputs=[user_id_input, movie_id_input, overview_input], outputs=output)
model.compile(loss='mse', optimizer='adam', metrics= ['acc']) 
model.summary()



# from sklearn.model_selection import train_test_split
# df_hybrid_train, df_hybrid_test = train_test_split(df_hybrid, test_size=0.2, random_state=42)

# X_train = [df_hybrid_train['Movie'].values, df_hybrid_train['User'].values, np.array([np.array(t) for t in df_hybrid_train['genre_feature']])]
# y_train = df_hybrid_train['Rating'].values

# X_test = [df_hybrid_test['Movie'].values, df_hybrid_test['User'].values, np.array([np.array(t) for t in df_hybrid_test['genre_feature']])]
# y_test = df_hybrid_test['Rating'].values


# df_hybrid_train = df_hybrid_train[['userId', 'Movie', 'genre_feature', 'scaler']]
# df_hybrid_train['genre_feature'] = df_hybrid_train['genre_feature'].tolist()
# df_hybrid_train['genre_feature'] = list(map(float, df_hybrid_train['genre_feature']))
# df_hybrid_train['genre_feature'] = list(map(int, df_hybrid_train['genre_feature']))



# Train and test the network
model.fit([df_hybrid_train['userId'].values, df_hybrid_train['Movieid'].values, np.array([np.array(t) for t in df_hybrid_train['overview']])],
          df_hybrid_train['rating'].values,
          batch_size=256,
          epochs=10,
        #   validation_split=0.2,
          verbose=1)

# # Train and test the network
# model.fit([df_hybrid_train['userId'].values, df_hybrid_train['Movie'].values, df_hybrid_train['Movie'].values, df_hybrid_train['Movie'].values, np.array([np.array(t) for t in df_hybrid_train['genre_feature']])],
#           df_hybrid_train['scaler'].values,
#           batch_size=256,
#           epochs=10,
#           validation_split=0.2,
#           verbose=1)

# # Train and test the network
# model.fit(X_train, y_train,
#           batch_size=64,
#           epochs=3,
#           validation_split=0.2,
#           shuffle=True,
#           verbose=1)



# model.save('/home/juno/workspace/user_collaborative_filtering/model_save/best_model_history/movie_metadata_genrehybridminmax_2m_ZSCORE_normalization4.h5')
# np.save('/home/juno/workspace/user_collaborative_filtering/model_save/best_model_history/hybrid_deeplearning_movie3.npy', model.history)
# model = tf.keras.models.load_model('/home/juno/workspace/user_collaborative_filtering/model_save/best_model_history/movie_metadata_genrehybridminmax_2m_ZSCORE_normalization4.h5')


## 예상치 추정 & csv 저장
y_pred = model.predict([df_hybrid_test['userId'].values, df_hybrid_test['Movie'].values, np.array([np.array(t) for t in df_hybrid_test['genre_feature']])])
y_pred = np.where(y_pred < -3.0795078415920294, -3.0795078415920294, np.where(y_pred > 1.4293505608665775, 1.4293505608665775, y_pred))
y_true = df_hybrid_test['scaler'].values


test = y_pred.reshape(400000,)
tmp = np.stack((y_true, test), axis=1)
print(tmp)
df_v = pd.DataFrame(tmp, columns=['true', 'pred'])
df_v.head(10)

result = df_v.sort_values(by=['true', 'pred'])
result =result.reset_index(drop=True)
# result.to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/recommend_data/metadata_genre_2m_pred_true.csv')/


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
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