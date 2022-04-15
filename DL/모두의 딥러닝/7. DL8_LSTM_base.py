# LSTM = Long Short Term Memory
# RNN
# 로이터 뉴스 카테고리 분석하기

from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.utils import np_utils

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# seed 값 고정하기
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 train, test set으로 나누기
# num_words = 1000 -> 1~1000개의 단어만 가져올 것
# test_split : train 80%, test 20%
(x_train, y_train), (x_test,y_test) = reuters.load_data(num_words=1000, test_split=0.2)

# 데이터 확인
category = np.max(y_train) + 1

# 데이터 전처리
# pad_sequence(maxlen) = 단어 수를 100개로 맞춰라 
# 단어 > 100 이라면 나머지 버림
# 단어 < 100 이라면 나머지 0으로 채움
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)

# One-Hot Encoding
y_trian = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 딥러닝 모델 설정
# Embedding =?
model = Sequential()
model.add(Embedding(1000, 100))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))
# model.add(Dense(1, activation='sigmoid'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 실행
history = model.fit(x_train, y_train, epochs=20, batch_size=50, validation_data=(x_test, y_test))

print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))

# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()