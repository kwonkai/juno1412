from keras.model import sequential
from keras.layers import dense

import numpy
import tensorflow as tf

# 실행 시 같은 결과 출력 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 준비된 수술환자 데이터 불러들이기
Data_set = numpy.loadtxt("C:/juno1412-1/DL/모두의 딥러닝/dataset/ThoraricSurgery.csv", delimiter=",")

# 환자기록, 수술결과 저장
X = Data_set[:, 0:17]
Y = Data_set[:, 17]

# 딥러닝 구조 설정
model = sequential()
model.add(Dense(30, input_dim = 17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 실행
model.compile(loss='mean_squared_error', optimize='adam')
metrics=(['accuracy'])
model.fit(X, Y, epoocho=30, batch_size=0)
