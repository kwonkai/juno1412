# 1. 라이브러리 설정
import pandas as pd
import numpy as np

# 시드값 고정
import tensorflow as tf
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# 2. 데이터 불러오기
yclass = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/y_class_콘텐츠 메타데이터/content_list_new.csv', header= 'infer')

from ast import literal_eval 
metadata_actor = yclass[['id','등장인물1','등장인물2']]
metadata_actor['등장인물'] = metadata_actor[['등장인물1','등장인물2']].fillna('').apply(lambda x:','.join(x), axis=1)

yclass = pd.merge(yclass, metadata_actor, on='id').drop(columns=['등장인물1_x', '등장인물2_x','등장인물1_y', '등장인물2_y'], axis=1)


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



drama_ids = []
comedy_ids = []
kpop_ids = []
others_ids = []

drama_t_cnt = 252
comedy_t_cnt = 187
kpop_t_cnt = 48
others_t_cnt = 24

drama_cnt = 0
comedy_cnt = 0
kpop_cnt = 0
others_cnt = 0

# 드라마 id 가져오기
for i in range(len(ml_movie_df_sort)):
    genre = ml_movie_df_sort['genre1'].iloc[i]
    if genre == 'Drama':
        if drama_cnt == drama_t_cnt:
            continue
        drama_ids.append(ml_movie_df_sort['movieId'].iloc[i])
        drama_cnt += 1 
    elif genre == 'Comedy':
        if comedy_cnt == comedy_t_cnt:
            continue
        comedy_ids.append(ml_movie_df_sort['movieId'].iloc[i])
        comedy_cnt += 1
        
    elif genre == 'Action':
        if kpop_cnt == kpop_t_cnt:
            continue
        kpop_ids.append(ml_movie_df_sort['movieId'].iloc[i])
        kpop_cnt += 1

    elif genre == 'Horror':     
        if others_cnt == others_t_cnt:
            continue
        others_ids.append(ml_movie_df_sort['movieId'].iloc[i])
        others_cnt += 1
        

    if drama_t_cnt==drama_cnt and comedy_t_cnt==comedy_cnt and kpop_t_cnt == kpop_cnt and others_t_cnt==others_cnt:
        break
    
rating = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/y_class_콘텐츠 메타데이터/movie_lens_official/ml-1m/ml-1m/ratings.dat', sep='::',header = None, engine='python',encoding='latin-1').drop(columns=[3], axis=1)
rating.columns = ['userId', 'movieId', 'rating']

drama_list = pd.DataFrame()
comedy_list = pd.DataFrame()
action_list = pd.DataFrame()
others_list = pd.DataFrame()

#for i in range(len(drama_ids)):
#    id = drama_ids[i]
#    drama_list.concat([drama_list, rating[rating['rating']==drama_ids]])
for i in range(len(drama_ids)):
    id = int(drama_ids[i])
    tmp = rating[rating['movieId']==id]
    if len(tmp) == 0:
        print("error:" + str(id))
    drama_list = pd.concat([drama_list, rating[rating['movieId']==id]])

for i in range(len(comedy_ids)):
    id = int(comedy_ids[i])
    tmp = rating[rating['movieId']==id]
    if len(tmp) == 0:
        print("error:" + str(id))
    comedy_list = pd.concat([comedy_list, rating[rating['movieId']==id]])


for i in range(len(kpop_ids)):
    id = int(kpop_ids[i])
    tmp = rating[rating['movieId']==id]
    if len(tmp) == 0:
        print("error:" + str(id))
    action_list = pd.concat([action_list, rating[rating['movieId']==id]])


for i in range(len(others_ids)):
    id = int(others_ids[i])
    tmp = rating[rating['movieId']==id]
    if len(tmp) == 0:
        print("error:" + str(id))
    others_list = pd.concat([others_list, rating[rating['movieId']==id]])



for i in range(len(others_ids)):
    other_list = others_ids    
    rating[rating['movieId']==others_ids[i]]

yclass_drama = []
yclass_comedy = []
yclass_kpop = []
yclass_others = []

yclass_drama = pd.DataFrame(yclass[yclass['장르']=='드라마'])
yclass_comedy = pd.DataFrame(yclass[yclass['장르']=='예능'])
yclass_kpop = pd.DataFrame(yclass[yclass['장르']=='K-pop'])
tmp1 = pd.DataFrame(yclass[yclass['장르']=='기타'])
tmp2 = pd.DataFrame(yclass[yclass['장르']=='인터뷰'])
tmp3 = pd.DataFrame(yclass[yclass['장르']=='강의'])
yclass_others = pd.DataFrame(pd.concat([tmp1, tmp2, tmp3]))


dic = {name:value for name,value in zip(drama_ids, yclass_drama['id'].tolist())}
drama_list = drama_list.replace({'movieId' : dic})

dic = {name:value for name,value in zip(comedy_ids, yclass_comedy['id'].tolist())}
comedy_list = comedy_list.replace({'movieId' : dic})

dic = {name:value for name,value in zip(kpop_ids, yclass_kpop['id'].tolist())}
action_list = action_list.replace({'movieId' : dic})

dic = {name:value for name,value in zip(others_ids, yclass_others['id'].tolist())}
others_list = others_list.replace({'movieId' : dic})

result_df = pd.concat([drama_list, comedy_list, action_list, others_list])

yclass = yclass.rename(columns={'id':'movieId'})
result_df = pd.merge(result_df, yclass, on='movieId')
result_df = result_df.sample(frac=1)

result_df.to_csv('/home/juno/workspace/user_collaborative_filtering/deeplearning_recomend/yclass_juno_rating_1m_20220825.csv')

'''

# ml_movie['movieId'] vs rating['movieId'] 비교 후, rating 0개 콘텐츠 뽑아내기
rating_list = rating['movieId'].tolist()
except_movie_df = ml_movie[~ml_movie['movieId'].isin(rating_list)]

# rating == 0인 콘텐츠 list만들기
except_movie_list = except_movie_df['movieId'].tolist()
# rating['movieId']에 없는지 다시 한번 확인
rating_except_check = rating[rating['movieId'].isin(except_movie_list)]

# ml_movie에서 rating 0인 콘텐츠 제거
ml_movie = ml_movie[~ml_movie['movieId'].isin(except_movie_list)]


drama_rating_df = pd.DataFrame()
'''
print('done')