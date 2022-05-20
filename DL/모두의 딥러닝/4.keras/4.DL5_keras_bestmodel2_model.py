# 베스트 모델 구하기
# 모델 설정하기
# 라이브러리 설정
from gc import callbacks
import os #추가
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.callback import ModelCheckpoint, EarlyStopping

# seed=0 고정설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


# 데이터 확인 및 분석
data_wine = pd.read_csv("C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/wine.csv", header=None)
data_wine
wine = data_wine.sample(frac=0.15) # frac =1 ? frac = 전체 row에서 몇%의 데이터를 return할 건지 설정


# 데이터 속성, 결론 값 dataset 설정하기
# 12개속성, 1개의 클래스
wine_df = wine.values
X = wine_df[:,0:12]
Y = wine_df[:, 12]

# 딥러닝 모델설정
model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', \
              optimizer='adam', \
              metrics=['accuracy'])

# 모델 설정
MODEL_DIR = 'C:/Users\kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/model/' # 저장 폴더 생성하기
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "C:/Users\kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/model/{epoch:02d}-{val_loss:4f}.hdf5"


# varbose =1 학습상황 보여주기
# safe_best_only = 모델이 앞서 저장한 모델보다 나아졌을 때만 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 모델 실행 및 저장
history = model.fit(X, Y, validation_split=0.33, epochs=1500, batch_size=500)

# y_vloss에 테스트셋으로 실험 결과의 오차 값을 저장
y_vloss=history.history['val_loss']

# y_acc 에 학습 셋으로 측정한 정확도의 값을 저장
y_acc=history.history['acc']

# x값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시
x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, "o", c="red", markersize=3)
plt.plot(x_len, y_acc, "o", c="blue", markersize=3)

plt.show()
