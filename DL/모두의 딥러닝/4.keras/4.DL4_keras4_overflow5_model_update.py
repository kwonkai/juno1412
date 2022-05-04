# 과적합 피하기
# k겹 교차검증 - kfold cross validation
# 라이브러리 설정
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 확인 및 시각화

data_wine = pd.read_csv("C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/wine.csv", header=None)
wine = data_wine.sample(frac=1) # frac =1 ?
wine.head(3)

# 12개속성, 1개의 클래스
wine_df = wine.values
X = wine_df[:,0:12]
Y = wine_df[:, 12]

# 딥러닝 설계
model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 컴파일
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=200, batch_size=100)


# 결과출력
print("\n Accuracy: %.4f" % (model.evaluate(X,Y)[1]))

