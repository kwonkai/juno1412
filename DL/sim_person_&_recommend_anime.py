# Commented out IPython magic to ensure Python compatibility.
# 라이브러리 설정
import random
import pandas as pd
import statistics
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.metrics.pairwise import cosine_similarity
import operator

# 데이터 불러오기

anime = pd.read_csv('/content/drive/MyDrive/anime/anime.csv')
rating = pd.read_csv('/content/drive/MyDrive/anime/rating.csv')

rating.shape

anime.shape

# 데이터 확인

rating.head(10)

anime.head(10)

# number of ratings
len(rating)

# number of users
len(rating['user_id'].unique())

# number of unique animes (in anime list, not ratings)
len(anime['anime_id'].unique())

# 콘텐츠를 평가한 유저 숫자

rating_per_user = rating.groupby('user_id')['rating'].count()
statistics.mean(rating_per_user.tolist())

# 유저별 평점 개수의 분포
rating_per_user.hist(bins=20, range=(0,500))

# 콘텐츠별 평가된 평점 등급
ratings_per_anime = rating.groupby('anime_id')['rating'].count()
statistics.mean(ratings_per_anime.tolist())

# 콘텐츠별 평가된 빈도
ratings_per_anime.hist(bins=30, range=(0,5000))

# rating(평가된 콘텐츠 전처리)
# 너무 적게 평가된 콘텐츠 제거
# counts of ratings per anime as a df
ratings_per_anime_df = pd.DataFrame(ratings_per_anime)

# 평점 개수가 100개 이하 콘텐츠 제거
filtered_ratings_per_anime_df = ratings_per_anime_df[ratings_per_anime_df.rating >= 500]
print(filtered_ratings_per_anime_df)

# df -> list
popular_anime = filtered_ratings_per_anime_df.index.tolist()

# anime(콘텐츠를 평가한 유저 데이터 처리)
# 너무 적게 평가한 유저 제거
# counts ratings per user as a df
rating_per_user_df = pd.DataFrame(rating_per_user)
# remove if < 500
filtered_rating_per_user_df = rating_per_user_df[rating_per_user_df.rating >= 200]
print(filtered_rating_per_user_df)
# build a list of user_ids to keep
prolific_users = filtered_rating_per_user_df.index.tolist()

filtered_rating = rating[rating.anime_id.isin(popular_anime)]
filtered_rating = rating[rating.user_id.isin(prolific_users)]
len(filtered_rating)

# 유저 리스트 만들기
user_id_list = list(set(filtered_rating['user_id']))
len(user_id_list)

# 피벗테이블 만들기
rating_matrix = filtered_rating.pivot_table(index='user_id', columns='anime_id', values='rating')

# 결측치 제거
rating_matrix = rating_matrix.fillna(0)
rating_matrix.head(15)

# 비슷한 유저 찾기
# 코드 분석
def similar_users(user_id, matrix, k=5):
    # 현재 유저에 대한 데이터프레임 만들기
    user = matrix[matrix.index == user_id]
    
    # matrix index 값이 user_id와 다른가?
    other_users = matrix[matrix.index != user_id]
    
    # 대상 user, 다른 유저와의 cosine 유사도 계산 
    # list 변환
    similarities = cosine_similarity(user,other_users)[0].tolist()
    
    # 다른 사용자의 인덱스 목록 생성
    indices = other_users.index.tolist()
    
    # 인덱스/유사도로 이뤄진 딕셔너리 생성
    index_similarity = dict(zip(indices, similarities))
    
    # 딕셔너리 정렬
    index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
    index_similarity_sorted.reverse()
    
    # 가장 높은 유사도 k개 정렬하기
    top_users_similarities = index_similarity_sorted[:k]
    users = [u[0] for u in top_users_similarities]
    
    return users

# 유저정보 list 뽑기
# 코드 제작 not 변수
random_user_id = random.choice(user_id_list)

# 콜  비슷한 유저
similar_user_indices = similar_users(random_user_id, rating_matrix)
print(similar_user_indices)


# 콘텐츠 추천
# 코드 분석
def recommend_item(user_index, similar_user_indices, matrix, items=10):
    
    # load vectors for similar users
    similar_users = matrix[matrix.index.isin(similar_user_indices)]
    # calc avg ratings across the 3 similar users
    similar_users = similar_users.mean(axis=0)
    # convert to dataframe so its easy to sort and filter
    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
    
    
    # load vector for the current user
    user_df = matrix[matrix.index == user_index]
    # transpose it so its easier to filter
    user_df_transposed = user_df.transpose()
    # rename the column as 'rating'
    user_df_transposed.columns = ['rating']
    # remove any rows without a 0 value. Anime not watched yet
    user_df_transposed = user_df_transposed[user_df_transposed['rating']==0]
    # generate a list of animes the user has not seen
    animes_unseen = user_df_transposed.index.tolist()
    
    # filter avg ratings of similar users for only anime the current user has not seen
    similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(animes_unseen)]
    # order the dataframe
    similar_users_df_ordered = similar_users_df.sort_values(by=['mean'], ascending=False)
    # grab the top n anime   
    top_n_anime = similar_users_df_ordered.head(items)
    top_n_anime_indices = top_n_anime.index.tolist()
    # lookup these anime in the other dataframe to find names
    anime_information = anime[anime['anime_id'].isin(top_n_anime_indices)]
    
    return anime_information #items

# 유저정보 list 뽑기
# 유저정보 뽑기
random_user_id = random.choice(user_id_list)
print(random_user_id)
# recommend_user = ""


# 추천 아이템 뽑아내기
recommend_item(random_user_id, similar_user_indices, rating_matrix)