# 라이브러리 설정
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy
import tensorflow as tf
tf.__version__

# 데이터 확인 및 시각화
# pandas로 읽어올 때, 순수 데이터 확인용
# 데이터에 header가 없어 names로 각 속성별 이름 붙여줌
data = pd.read_csv('C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/pima-indians-diabetes.csv', names = ["pregnant", "plasma", "pressure", "thickness", "insulin", "bmi", "pedigree", "age", "class"])
print(data.head(5))

# 데이터 전처리
# 속성별 정보의 당뇨병 발병 상관관계 계산
# as_type = pregnant정보 옆 새로운 index 생성
# sort_values(by='pregnant', ascending=True -> pregnant 기준 오름차순 정렬
print(data[['pregnant','class']].groupby(['pregnant'], as_index=False).mean().sort_values(by='pregnant', ascending=True))

# 데이터 시각화
# vmax = 색상의 밝기, cmap = matplotlib의 색상 설정값 부여
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), linewidths=0.1, vmax=0.5, vmin=-0.5, cmap='RdYlBu_r', linecolor='white', annot=True )
plt.show()

grid = sns.FacetGrid(data, col='class')
grid.map(plt.hist, 'plasma', bins=10)
plt.show()

# 피마인디언 당뇨병 예측 실행
# seed=0 설정 -> 실행할 때마다 같은 결과 출력
seed=0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 수술 환자 데이터 불러오기
# 분리, delimiter=","
data_set = numpy.loadtxt('C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/pima-indians-diabetes.csv', delimiter=",")

# 사람의 속성, 당뇨병여부(클래스) 저장
x = data_set[:,0:8] # 속성
y = data_set[:,8] # 클래스

# 딥러닝 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 실행
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=10)

# 출력
print("\n Accuracy: %.4f" % (model.evaluate(x, y)[1]))