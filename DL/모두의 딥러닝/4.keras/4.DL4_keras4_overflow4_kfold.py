# 과적합 피하기
# k겹 교차검증 - kfold cross validation
# 라이브러리 설정
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
import tensorflow as tf

# seed 설정
seed=0
np.random.seed(seed)
tf.set_random_seed(seed)


# 데이터 확인 및 분석
data = pd.read_csv("C:/Users/kwonk/juno1412-1/juno1412/DL/모두의 딥러닝/dataset/sonar.csv", header=None)

dataset = data.values
X = dataset[:, 0:60]
Y_obj = dataset[:, 60]

# 문자열 변환
# 라벨 인코딩 -> 원-핫 인코딩
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# k겹 교차검증-kfold cross validation
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

# 빈 accuracy 배열
accuracy = []

# 모델 설정, 컴파일, 실행
for train, test in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])
    # # 딥러닝 fit, 모델 실행
    # x[train], y[train] = X, Y 값에서 분리하여 TRAIN, TEST SET으로 이용
    model.fit(X[train], Y[train], epochs=200, batch_size = 5) 
    k_accuracy = "%.4f" % (model.evaluate(X[test], Y[test])[1])
    accuracy.append(k_accuracy)

print("\n %.f fold accuracy:" % n_fold, accuracy)



