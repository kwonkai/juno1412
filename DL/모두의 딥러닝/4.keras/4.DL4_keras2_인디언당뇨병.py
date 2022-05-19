# keras 함수 불러오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 라이브러리 불러오기
import numpy
import tensorflow as tf
import pandas as pd

# seed=0 설정 -> 실행할 때마다 같은 결과 출력
seed=0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
data_set = pd.read_csv('C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/pima-indians-diabetes.csv', names=["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])
data_set
x = data_set[:,0:8] # 속성 = 사람 특성
y = data_set[:,8] # 클래스 =  당뇨병 여부

# 상관도 그래프
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.pairplot(data_set, hue="class")
# plt.figure(figsize=(10,5))
# plt.show()

# 딥러닝 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # 속성 8개에 relu 2번은 이해
model.add(Dense(8, activation="relu")) # 왜 relu 2번?
model.add(Dense(1, activation="sigmoid")) # 출력만 sigmoid, 

# 딥러닝 모델  컴파일
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 실행
model.fit(x, y, epochs=250, batch_size = 10)

# 결과출력
print("\n Accuracy: %.4f" % (model.evaluate(x, y)[1]))