# 다중분류 문제 해결
# 라이브러리 설정
from typing import Sequence
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils import np_utils

#seed 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 입력
data = pd.read_csv("C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/iris.csv", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

# 상관도 그래프
sns.pairplot(data, hue='species');
# plt.show()

# 데이터셋
dataset = data.values
X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4]

# 원핫 인코딩
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = np_utils.to_categorical(Y)

# softmax 활성화함수
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation="softmax"))

# model compile
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X, Y_encoded, epochs=60, batch_size = 5)

print('\n Accuracy: %.4f' % (model.evaluate(X, Y_encoded)[1]))





