import numpy as np
import pandas as pd
import surprise
from surprise import Dataset
from surprise import SVD
from surprise.model_selection import train_test_split

# ml-100k: 10만 개 평점 데이터
data = Dataset.load_builtin('ml-100k')

# surprise의 train_test_split() 사용
trainset, testset = train_test_split(data, test_size=0.25, random_state=0)

# SVD를 이용한 잠재 요인 협업 필터링
algo = SVD()
algo.fit(trainset)

# 사용자 아이디(uid), 아이템 아이디(iid)는 문자열로 입력
uid = str(52)
iid = str(305)

pred = algo.predict(uid, iid)
pred

predictions = algo.test( testset )
print('prediction type :',type(predictions), ' size:',len(predictions))
print('prediction 결과의 최초 5개 추출')


predictions[:5]

# 성능 평가
from surprise import accuracy

accuracy.rmse(predictions)


# index와 header를 제거한 ratings_noh.csv 파일 생성
info = pd.read_csv("C:/kwon/firstkino-data/data-files/movie_info_result.csv")
info.to_csv('C:/kwon/firstkino-data/data-files/movie_info_result_noh.csv', index=False, header=False)


# Reader 객체 생성
from surprise import Reader
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
data = Dataset.load_from_file('C:/kwon/firstkino-data/data-files/movie_info_result_noh.csv', reader=reader)
data


# surprise의 train_test_split() 사용
trainset, testset = train_test_split(data, test_size=0.25, random_state=0)

# SVD를 이용한 잠재 요인 협업 필터링 (잠재 요인 크기 = 50)
algo = SVD(n_factors=50, random_state=0)
algo.fit(trainset)

# 추천 예측 평점 (.test)
predictions = algo.test( testset )

# 성능 평가
accuracy.rmse(predictions)



from surprise.model_selection import cross_validate 

# 데이터 불러오기 (데이터 프레임)
ratings = pd.read_csv('C:/kwon/firstkino-data/data-files/movie_info_result.csv') 

# Reader 객체 생성
reader = Reader(rating_scale=(0.5, 5.0))

# 사용자 아이디, 아이템 아이디, 평점 순서 (원래는 timestamp도 있으나 제외)
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# SVD를 이용한 잠재 요인 협업 필터링
algo = SVD(random_state=0)

# 교차 검증 수행
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

from sklearn.model_selection import HalvingGridSearchCV

# n_epochs: SGD 수행 시 반복 횟수, n_factors: 잠재 요인 크기
param_grid = {
    'n_epochs': [20, 40, 60], 
    'n_factors': [50, 100, 200]
}

gs = HalvingGridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3) # algo가 아닌 SVD 입력하였다.
gs.fit(data)



