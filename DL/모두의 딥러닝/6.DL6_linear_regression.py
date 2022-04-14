# 선형회귀

# 라이브러리 설정
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 확인 및 분석
data = pd.read_csv("C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/housing.csv", delim_whitespace=True, header=None)

print(data.info())


# 데이터셋 설정
dataset = data.values # data.values = index를 제외한 나머지 칼럼들의 값
X = dataset[:, 0:13]
Y = dataset[:, 13]

# 훈련셋, 테스트셋 구분
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)

# 딥러닝 모델 구축
model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1)) # 선형회귀 데이터는 마지막에 참/거짓 구분 불필요

# 딥러닝 모델 컴파일 & 실행
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size = 10)

# 결론 도출
# 실제값 VS 예측값
Y_prediction = model.predict(x_test).flatten()
for i in range(10):
    label = y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))

