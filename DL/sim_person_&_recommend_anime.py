# Commented out IPython magic to ensure Python compatibility.
# 1. 라이브러리 설정
import random
import pandas as pd
import statistics
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.metrics.pairwise import cosine_similarity
import operator

# 2. 데이터 불러오기 & 확인 & 시각화

anime = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/anime/anime.csv')
rating = pd.read_csv('/home/juno/workspace/user_collaborative_filtering/data_files/anime/rating.csv')



# 데이터 숫자/고유값
len(rating)
# number of users
len(rating['user_id'].unique())
# number of unique animes (in anime list, not ratings)
len(anime['anime_id'].unique())


# 데이터 시각화
# 콘텐츠를 평가한 유저 숫자
rating_per_user = rating.groupby('user_id')['rating'].count()
statistics.mean(rating_per_user.tolist()) # 1인당 106.28765558049378

# 유저별 평점 개수의 분포
rating_per_user.hist(bins=20, range=(0,500))

# 콘텐츠별 평가된 평점 등급
ratings_per_anime = rating.groupby('anime_id')['rating'].count()
statistics.mean(ratings_per_anime.tolist()) # 1콘텐츠당 697.3번 리뷰

# 콘텐츠별 평가된 빈도
ratings_per_anime.hist(bins=30, range=(0,5000))




# 3. 데이터 전처리
# rating 평점 '-1' -> '5'으로 변경 = normal
# -1 : 보았으나 평가하지 않은 데이터
rating = rating.replace({'rating':-1},5)


# rating(평가된 콘텐츠 전처리)
# 너무 적게 평가된 콘텐츠 제거(50개 이하 평가 데이터 제거)
ratings_per_anime_df = pd.DataFrame(ratings_per_anime)

filtered_ratings_per_anime_df = ratings_per_anime_df[ratings_per_anime_df.rating >= 50]
popular_anime = filtered_ratings_per_anime_df.index.tolist()
len(filtered_ratings_per_anime_df)

# anime(콘텐츠를 평가한 유저 데이터 처리)
# 너무 적게 평가한 유저 제거 -> 30개 이하 평가한 유저 제거
rating_per_user_df = pd.DataFrame(rating_per_user)
filtered_rating_per_user_df = rating_per_user_df[rating_per_user_df.rating >= 30]
rating_users = filtered_rating_per_user_df.index.tolist()
len(filtered_rating_per_user_df)

filtered_rating = rating[rating.anime_id.isin(popular_anime)]
filtered_rating = rating[rating.user_id.isin(rating_users)]
len(filtered_rating)

# 유저 리스트 만들기 # 랜덤 유저 뽑아내는 용도
user_id_list = list(set(filtered_rating['user_id']))




# 4. 피벗테이블 만들기
rating_matrix = filtered_rating.pivot_table(index='user_id', columns='anime_id', values='rating')

# 결측치 제거
rating_matrix = rating_matrix.fillna(0)



# 5. 비슷한 유저 찾기
def similar_users(user_id, matrix, k=5):
    # 현재 유저에 대한 정보찾기
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
# # 코드 제작 not 변수 -> input or user searching 함수 제작
# random_user_id = random.choice(user_id_list)


# 특정 유저에게 콘텐츠 추천
recommend_user = 315

# # call similarity user
# similar_user_indices = similar_users(random_user_id, rating_matrix)

# call similarity user
similar_user_indices = similar_users(recommend_user, rating_matrix)



# 7. 콘텐츠 추천
# 코드 분석
def recommend_item(user_index, similar_user_indices, matrix, items=10):
    # 유저와 비슷한 유저 가져오기
    similar_users = matrix[matrix.index.isin(similar_user_indices)]
    # 비슷한 유저 평균 계산 # row 계산
    similar_users = similar_users.mean(axis=0)
    # dataframe 변환 : 정렬/필터링 용이
    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])


    # 현재 사용자의 벡터 가져오기 : matrix = rating_matrix(pivot table)
    user_df = matrix[matrix.index == user_index]

    # 현재 사용자의 평가 데이터 정렬
    user_df_transposed = user_df.transpose()

    # 컬럼명 변경 user_id : 48432 -> rating
    user_df_transposed.columns = ['rating']

    # 미시청 콘텐츠는 rating 0로 바꾸어 준다. remove any rows without a 0 value. Anime not watched yet
    user_df_transposed = user_df_transposed[user_df_transposed['rating']==0]

    # 미시청 콘텐츠 목록리스트 만들기
    animes_unseen = user_df_transposed.index.tolist()

    # 안본 콘텐츠 필터링
    similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(animes_unseen)]

    # 평균값을 내림차순 정렬
    # sort_values(by=[정렬 기준 축])
    # 'mean' = similar_user의 columns의 이름
    similar_users_df_ordered = similar_users_df_filtered.sort_values(by=['mean'], ascending=False)

    # 상위 10개 값 가져오기
    # items = 10
    top_n_anime = similar_users_df_ordered.head(items)
    top_n_anime_indices = top_n_anime.index.tolist()

    # anime dataframe에서 top10값 찾기
    anime_information = anime[anime['anime_id'].isin(top_n_anime_indices)]
    anime_information
    
    return anime_information #items



# 유저정보 list 뽑기
# 유저정보 뽑기
# random_user_id = random.choice(user_id_list)
# print(random_user_id)

# 추천 콘텐츠 뽑아내기 # 랜덤
# recommend_content = recommend_item(random_user_id, similar_user_indices, rating_matrix)



# 특정 유저에게 지정
# recommend_user = 191

# 추천 콘텐츠 뽑아내기 #특정 유저
recommend_content = recommend_item(recommend_user, similar_user_indices, rating_matrix)

print("-- 콘텐츠 추천 TOP 10 --")

# 모든 추천
print(recommend_content)
print("===================================")

# movie_id만 뽑기
print("-- ID --")
print(recommend_content['anime_id'])
print("===================================")

# name만 뽑기
print("-- 제목 --")
print(recommend_content['name'])
print("===================================")

# rating만 뽑기
print("-- 평점 --")
print(recommend_content['name'])
print("===================================")
# 특정유저에게 콘텐츠 추천학시
# recommend_content = recommend_item(recommend_user, similar_user_indices, rating_matrix)
# print(recommend_content)
