import pandas as pd
from pyparsing import col
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate 
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import GridSearchCV

import os

# 데이터 불러오기
ratings = pd.read_csv("추천시스템/ratings.csv")
ratings.to_csv("추천시스템/ratings_clean.csv", index=False, header=False)

# reader, data 설정
col = 'userId movieId rating'
reader = Reader(line_format=col, sep=',', rating_scale=(1, 5))
data=Dataset.load_from_file('추천시스템/ratings_clean.csv', reader=reader)

trainset, testset = train_test_split(data, test_size=.25, random_state=42) # 수행 시마다 동일한 결과를 도출하기 위해 random_state 설정

algo = SVD(n_factors=50, random_state=42)

# 학습 데이터 세트로 학습하고 나서 테스트 데이터 세트로 평점 예측 후 RMSE 평가
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# 교차검증
# 교차 검증 수행
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# n_epochs: SGD 수행 시 반복 횟수, n_factors: 잠재 요인 크기
param_grid = {
    'n_epochs': [20, 40, 60], 
    'n_factors': [50, 100, 200]
}

# GridSearchCV
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

print(gs.best_params['rmse'])
print(gs.best_score['rmse'])

# 사용자 추천 예측평점
uid = str(9)
iid = str(42)
pred = algo.predict(uid, iid, verbose=True)

movies = pd.read_csv('C:/Users/kwonk/Downloads/개인 프로젝트/juno1412-1/추천시스템/movies.csv')

# 아직 보지 않은 영화 리스트 함수
def get_unseen_surprise(ratings, movies, userId):
     # 특정 userId가 평점을 매긴 모든 영화 리스트
    seen_movies = ratings[ratings['userId']== userId]['movieId'].tolist()
    
    # 모든 영화명을 list 객체로 만듬. 
    total_movies = movies['movieId'].tolist()
      
    # 한줄 for + if문으로 안 본 영화 리스트 생성
    unseen_movies = [ movie for movie in total_movies if movie not in seen_movies]
    
    # 일부 정보 출력
    total_movie_cnt = len(total_movies)
    seen_cnt = len(seen_movies)
    unseen_cnt = len(unseen_movies)
    
    print(f"전체 영화 수: {total_movie_cnt}, 평점 매긴 영화 수: {seen_cnt}, 추천 대상 영화 수: {unseen_cnt}")
    
    return unseen_movies

unseen_movies = get_unseen_surprise(ratings, movies, 115)

def recomm_movie_by_surprise(algo, userId, unseen_movies, top_n=10):
    
    # 아직 보지 않은 영화의 예측 평점: prediction 객체 생성
    predictions = []    
    for movieId in unseen_movies:
        predictions.append(algo.predict(str(userId), str(movieId)))
    
    # 리스트 내의 prediction 객체의 est를 기준으로 내림차순 정렬
    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True) # key에 리스트 내 객체의 정렬 기준을 입력
    
    # 상위 top_n개의 prediction 객체
    top_predictions = predictions[:top_n]
    
    # 영화 아이디, 제목, 예측 평점 출력
    print(f"Top-{top_n} 추천 영화 리스트")
    
    for pred in top_predictions:
        
        movie_id = int(pred.iid)
        movie_title = movies[movies["movieId"] == movie_id]["title"].tolist()
        movie_rating = pred.est
        
        print(f"{movie_title}: {movie_rating:.2f}")

recomm_movie_by_surprise(algo, 115, unseen_movies, top_n=10)


