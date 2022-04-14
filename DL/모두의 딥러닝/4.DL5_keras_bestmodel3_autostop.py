# 학습의 자동중단

# 라이브러리 설정
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping



# seed 고정 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 가져오기 & 데이터셋 설정
data = pd.read_csv("C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/wine.csv", header=None)
dataset = data.values
X = dataset[:,0:12]
Y = dataset[:,12]


# 딥러닝 모델 설정

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 컴파일 & 실행
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 자동중단 설정
# 오차가 100 이상에 도달하면 자동으로 중단한다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience = 100)

model.fit(X, Y, validation_split=0.2, epochs=3000, batch_size=500, callbacks=[early_stopping_callback])

# 결과출력
print("\n Accuracy: %.4f" % (model.evaluate(X,Y)[1]))