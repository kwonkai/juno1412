# keras 함수 불러오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 라이브러리 불러오기
import numpy
import tensorflow as tf
tf.__version__

# seed=0 설정 -> 실행할 때마다 같은 결과 출력
seed=0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 수술 환자 데이터 불러오기
# 분리, delimiter=","
data_set = numpy.loadtxt('C:/Users/kwonk/Downloads/개인 프로젝트/juno1412-1/DL/모두의 딥러닝/dataset/ThoraricSurgery.csv', delimiter=",")

# 환자의기록, 수술 결과 저장
x = data_set[:,0:17] # 속성
y = data_set[:,17] # 클래스

# 딥러닝 모델 설정
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 실행
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=30, batch_size=10)

# 출력
print("\n Accuracy: %.4f" % (model.evaluate(x, y)[1]))