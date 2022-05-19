# 다중분류 문제 해결
# 라이브러리 설정
from typing import Sequence
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

from sklearn.model_selection import train_test_split
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
data.head(10)
# 상관도 그래프
sns.pairplot(data, hue='species');
plt.show()

# 데이터셋
dataset = data.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4] # class : iris 속성값

# 원핫 인코딩

e = LabelEncoder() # 각 class 문자열 -> 숫자로 변환
e.fit(Y)
Y_one = e.transform(Y)
Y_encoded = np_utils.to_categorical(Y_one)  # label을 원핫인코딩(0,1)으로 변경 -> relu 활성화함수에 맞게 변경



# x_test, y_test 값 도출
# softmax 활성화함수


model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation="softmax")) # 다중분류이므로 출력층 node 수 3개

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(X, Y_encoded, epochs=60, batch_size = 5)

print('\n Accuracy: %.4f' % (model.evaluate(X, Y_encoded)[1]))



# x_test, y_test
# 실제값 적용해보기
# softmax 활성화함수

model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation="softmax")) # 다중분류이므로 출력층 node 수 3개

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(X, Y_encoded, epochs=60, batch_size = 5)

print('\n Accuracy: %.4f' % (model.evaluate(x_test, y_test)[1]))


# # 은닉층 작을 때

# model = Sequential()
# model.add(Dense(16, input_dim=4, activation='relu'))
# model.add(Dense(1, activation='relu'))
# model.add(Dense(3, activation="softmax")) # 다중분류이므로 출력층 node 수 3개

# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
# model.fit(X, Y_encoded, epochs=60, batch_size = 5)

# print('\n Accuracy: %.4f' % (model.evaluate(X, Y_encoded)[1]))



