# keras 함수, 라이브러리 불러오기
from gc import callbacks
import os
from tabnanny import check
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# 실행 시 같은 결과 출력 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# 준비된 수술환자 데이터 불러들이기
Data_set = np.loadtxt("C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/ThoraricSurgery.csv", delimiter=",")

# 환자기록, 수술결과 저장
# X = 환자 상태, Y = 생존여부
X = Data_set[:, 0:17] 
Y = Data_set[:, 17]

# 딥러닝 구조 설정
model = Sequential()
model.add(Dense(30, input_dim = 17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# 모델 저장경로&폴더 생성
MODEL_DIR = 'C:/Users\kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/model/' 
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR) 

modelpath = "C:/Users\kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/model/DL1.hdf5"

# 체크포인트 모니터 = loss 값
from tensorflow.keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='loss',  verbose=1)


# 모델 훈련 & 결과 출력
history = model.fit(X, Y, epochs=30, batch_size=50, verbose=1, callbacks=[checkpointer])
print("\n Accuracy : %.4f" %(model.evaluate(X,Y)[1]))


# 그래프 결과값 시각화
y_vloss = history.history['loss']
y_acc = history.history['acc']

x_len1 = np.arange(len(y_acc))
# plt.plot(x_len1, y_vloss, "o", c="red")
b_plot = plt.plot(x_len1, y_acc, "o", c="blue")

plt.xlabel("epoch")
plt.ylabel("accuracy")

plt.show()