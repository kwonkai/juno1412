# 과적합 피하기
# 라이브러리 설정
from tkinter import Y, Label
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

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

# 딥러닝 모델 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 모델 컴파일
model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])

# 딥러닝 모델 실행
model.fit(X, Y, epochs=200, batch_size = 5)

print("\n Accuracy: %4f" % (model.evaluate(X,Y)[1]))