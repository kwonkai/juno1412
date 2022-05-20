# LSTM = Long Short Term Memory
# RNN & CNN
# IMDB 데이터 활용
# 라이브러리 설정하기
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# seed 값 고정하기
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 train, test set으로 나누기
# num_words = 5000 -> 1~5000개의 단어만 가져올 것
# test_split : train 80%, test 20%
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

# 데이터 전처리
# pad_sequence(maxlen) = 단어 수를 100개로 맞춰라 
# 단어 > 100 이라면 나머지 버림
# 단어 < 100 이라면 나머지 0으로 채움
x_train = sequence.pad_sequences(x_train, maxlen = 100)
x_test = sequence.pad_sequences(x_test, maxlen = 100)


# 딥러닝 모델 설정
model = Sequential()
model.add(Embedding(5000,100))
model.add(Dropout(0.5))
model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()


model.compile(loss ='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 실행
history = model.fit(x_train, y_train, epochs=20, batch_size = 200, validation_data=(x_test, y_test))

print("\n Train Accuracy: %.4f" % (model.evaluate(x_train, y_train)[1]))
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
