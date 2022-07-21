import pandas as pd
import numpy as np
from ast import literal_eval


from sklearn.preprocessing import MultiLabelBinarizer

movie_metadata = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/netflix/movies_metadata.csv', low_memory=False)[['id', 'original_title','genres', 'production_countries', 'vote_count']].drop([19730, 29503, 35587], axis=0)
movie_metadata['id'] = movie_metadata['id'].astype(float).astype(int)
########### 장르 전처리 #######

# genre id/name 해체
from ast import literal_eval # 문자 가공해서 파이썬 객체생성해줌 // 문자그대로 평가
movie_metadata['genres'] = movie_metadata['genres'].apply(literal_eval)
movie_metadata['genre_name'] = movie_metadata['genres'].apply(lambda x : [ y['name'] for y in x])



# gener_name columns 에서 list화 해제하기


## 방송국 지우기
genre_feature = pd.DataFrame(movie_metadata['genre_name'].to_list())
genre_feature.columns = ['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8']

genre_feature = genre_feature.drop(columns=['genre6', 'genre7', 'genre8'], axis=1).fillna('')


genre_feature['genre_feature'] = genre_feature[['genre1', 'genre2', 'genre3', 'genre4', 'genre5']].apply(lambda x: ','.join(x), axis=1)



remove_list = ['Carousel Productions', 'Vision View Entertainment', 'Telescene Film Group Productions',\
               'Aniplex', 'GoHands', 'BROSTA TV', 'Mardock Scramble Production Committee', 'Sentai Filmworks', \
               'Odyssey Media', 'Pulser Productions', 'Rogue State', 'The Cartel'] 

genre_feature = genre_feature[~genre_feature['genre_feature'].isin(remove_list)].reset_index().rename(columns={'index':'id'})
movie_metadata = pd.merge(movie_metadata, genre_feature, on = 'id').drop(columns=['genres', 'genre_name']).drop(columns=['genre1', 'genre2', 'genre3', 'genre4', 'genre5'],axis=1)

## 장르 0, 1 값으로 변경하기
mat = movie_metadata.to_numpy()
genres = mat[:,4]

mlb = MultiLabelBinarizer()
mlb.fit(genres)
movie_metadata['genre_feature'] = movie_metadata['genre_feature'].apply(lambda x: mlb.transform([x])[0])

## rating 데이터 불러오기
rating = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/netflix/ratings (2).csv').rename(columns={"movieId" : "id"})

hybrid_df = pd.merge(rating, movie_metadata, on='id').to_csv('/home/juno/workspace/user_collaborative_filtering/data_files/deeplearning/genre_hybrid_df.csv')

