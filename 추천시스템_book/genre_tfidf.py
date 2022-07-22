import pandas as pd
import numpy as np
from ast import literal_eval

from sklearn.preprocessing import MultiLabelBinarizer

# To compute similarities between vectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# To create deep learning models
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model

# 평균값 구하기
import statistics

# To stack sparse matrices
from scipy.sparse import vstack

### 데이터 불러오기 & 잘못들어가 있는 값 제거
# id 대신 date값이 들어가있는 index 제거
movie_metadata = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/netflix/movies_metadata.csv', low_memory=False)
movie_metadata = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/netflix/movies_metadata.csv', low_memory=False)[['id', 'original_title','genres', 'production_countries', 'vote_count']].drop([19730, 29503, 35587], axis=0)
# id 컬럼 type 변경
movie_metadata['id'] = movie_metadata['id'].astype(float).astype(int)


########### 장르 전처리 #######
# genre id/name 해체
from ast import literal_eval # 문자 가공해서 파이썬 객체생성해줌 // 문자그대로 평가
movie_metadata['genres'] = movie_metadata['genres'].apply(literal_eval)
movie_metadata['genre_name'] = movie_metadata['genres'].apply(lambda x : [ y['name'] for y in x])


## 중첩리스트 제거 & 방송국 지우기
genre_feature = pd.DataFrame(movie_metadata['genre_name'].to_list())
genre_feature.columns = ['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8']

remove_list = genre_feature.isin(['Carousel Productions', 'Vision View Entertainment', 'Telescene Film Group Productions',\
               'Aniplex', 'GoHands', 'BROSTA TV', 'Mardock Scramble Production Committee', 'Sentai Filmworks', \
               'Odyssey Media', 'Pulser Productions', 'Rogue State', 'The Cartel'])

genre_feature = genre_feature[~remove_list]   


# genre 유일값 확인하기 : 방송국 제거되었는지 확인
print(pd.unique(genre_feature[['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8']].values.ravel()))


genre_feature['genre_feature'] = genre_feature.fillna('').apply(lambda x: ','.join(x), axis=1)
genre_feature = genre_feature.reset_index().rename(columns={'index':'id'})
remove_list = ['Carousel Productions', 'Vision View Entertainment', 'Telescene Film Group Productions',\
               'Aniplex', 'GoHands', 'BROSTA TV', 'Mardock Scramble Production Committee', 'Sentai Filmworks', \
               'Odyssey Media', 'Pulser Productions', 'Rogue State', 'The Cartel'] 

movie_metadata = pd.merge(movie_metadata, genre_feature, on = 'id').drop(columns=['genres', 'genre_name']).drop(columns=['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8'],axis=1)

## 장르 0, 1 값으로 변경하기
mat = movie_metadata
genres = mat[:,5]

mlb = MultiLabelBinarizer()
mlb.fit(genres)
movie_metadata['genre_feature'] = movie_metadata['genre_feature'].apply(lambda x: mlb.transform([x])[0])

## rating 데이터 불러오기 & id기준으로 병합하기
rating = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/netflix/ratings.csv').rename(columns={"movieId" : "id"})

hybrid_df = pd.merge(rating, movie_metadata, on='id')
# np.save('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/genre_hybrid_df.npy', hybrid_df)


### hybrid dataframe 유효값 전처리
# 1) 콘텐츠에 평가된 개수 유효값 처리
hybrid_df = hybrid_df[hybrid_df['vote_count']>1130]
statistics.mean(hybrid_df.vote_count.tolist())

# 2) 유저가 평가한 콘텐츠 개수 유효값 처리
rating_count = rating.groupby('userId')['rating'].count()
statistics.mean(rating_count.tolist())

rating_count_df = pd.DataFrame(rating_count)
filter_rating_count_df = rating_count_df[rating_count_df.rating > 100]
hybrid_df_filter = hybrid_df[hybrid_df.userId.isin(filter_rating_count_df.reset_index().userId)]


## 중첩리스트 제거 & 방송국 지우기
genre_feature2 = pd.DataFrame(hybrid_df_filter['genre_feature'].to_list())
genre_feature2.columns = ['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8', 'genre9', 'genre10', 'genre11', 'genre12', 'genre13', 'genre14', 'genre15', \
                          'genre16', 'genre17', 'genre18', 'genre19', 'genre20', 'genre21', 'genre22', 'genre23', 'genre24', 'genre25', 'genre26', 'genre27', 'genre28', 'genre29', 'genre30']


genre_feature2['genre_feature2'] = genre_feature2.astype(str).apply(lambda x: ','.join(x), axis=1)
genre_feature2 = genre_feature2.reset_index().rename(columns={'index':'id'}).drop(columns=['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8', 'genre9', 'genre10', 'genre11', 'genre12', 'genre13', 'genre14', 'genre15', \
                          'genre16', 'genre17', 'genre18', 'genre19', 'genre20', 'genre21', 'genre22', 'genre23', 'genre24', 'genre25', 'genre26', 'genre27', 'genre28', 'genre29', 'genre30'],axis=1)
movie_metadata = pd.merge(hybrid_df_filter, genre_feature2, on = 'id').drop(columns=['genre_feature']).rename(columns={'genre_feature2':'genre_feature'})
movie_metadata
np.save('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/genre_hybrid_df_cleaning2.npy', movie_metadata)


hybrid_df_filter_genre = hybrid_df_filter['genre_feature'].toarray()
# ratings_per_anime_df = pd.DataFrame(ratings_per_anime)

# filtered_ratings_per_anime_df = ratings_per_anime_df[ratings_per_anime_df.rating >= 50]
# popular_anime = filtered_ratings_per_anime_df.index.tolist()

# # 콘텐츠를 평가한 유저 숫자
# rating_per_user = rating.groupby('user_id')['rating'].count()
# statistics.mean(rating_per_user.tolist()) # 1인당 106.28765558049378

# # 콘텐츠별 평가된 평점 등급
# ratings_per_anime = rating.groupby('anime_id')['rating'].count()
# statistics.mean(ratings_per_anime.tolist()) # 1콘텐츠당 697.3번 리뷰
# filtered_rating = rating[rating.anime_id.isin(popular_anime)]



# Create tf-idf matrix for text comparison
tfidf = CountVectorizer()
tfidf_hybrid = tfidf.fit_transform(hybrid_df_filter_genre)

# train_test_split
hybrid_df_train = tfidf_hybrid[150000:]
hybrid_df_test = tfidf_hybrid[:30000]


# Get mapping from movie-ids to indices in tfidf-matrix
mapping = {id:i for i, id in enumerate(hybrid_df_filter.index)}

train_tfidf = []
# Iterate over all movie-ids and save the tfidf-vector
for id in hybrid_df_train['genre_feature'].values:
    index = mapping[id]
    train_tfidf.append(tfidf_hybrid[index])
    
test_tfidf = []
# Iterate over all movie-ids and save the tfidf-vector
for id in hybrid_df_test['genre_feature'].values:
    index = mapping[id]
    test_tfidf.append(tfidf_hybrid[index])


# Stack the sparse matrices
train_tfidf = vstack(train_tfidf)
test_tfidf = vstack(test_tfidf)
train_tfidf = train_tfidf.toarray()
test_tfidf = test_tfidf.toarray()



# genre_list.index[genre_list['B'] == 19].tolist()/