# 과적합 피하기
# 모델 저장 및 재사용
# 라이브러리 설정
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow as tf

# seed 설정
seed=0
np.random.seed(seed)
tf.set_random_seed(seed)


# 데이터 확인 및 분석
data = pd.read_csv("C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/sonar.csv", header=None)

dataset = data.values
X = dataset[:, 0:60]
Y_obj = dataset[:, 60]

# 문자열 변환
# 라벨 인코딩 -> 원-핫 인코딩
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed) # 왜 seed? 초기화? 일정함?


# 딥러닝 모델 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# # 딥러닝 모델 컴파일
model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])

# # 딥러닝 fit, 모델 실행
model.fit(x_train, y_train, epochs=200, batch_size = 5)
model.save('C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/my_model.h5') #모델저장

# 모델 불러오기
del model # 테스트를 위해 메모리 내의 모델 삭제
model = load_model('C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/my_model.h5') # 모델 새로 불러오기


print("\n Accuracy: %4f" % (model.evaluate(x_test, y_test)[1]))