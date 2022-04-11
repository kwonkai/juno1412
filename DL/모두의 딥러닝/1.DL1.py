# keras 함수, 라이브러리 불러오기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf

# 실행 시 같은 결과 출력 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

# 준비된 수술환자 데이터 불러들이기
Data_set = numpy.loadtxt("C:/juno1412-1/DL/모두의 딥러닝/dataset/ThoraricSurgery.csv", delimiter=",")

# 환자기록, 수술결과 저장
X = Data_set[:, 0:17]
Y = Data_set[:, 17]

# 딥러닝 구조 설정
model = Sequential()
model.add(Dense(30, input_dim = 17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 실행
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=0)

# 결과출력
print("\n Accuracy : %.4f" %(model.evaluate(X,Y)[1]))