# 1.라이브러리 불러오기
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.layers import Dense, Concatenate, Activation

train_df = pd.read_csv(r'C:\Users\kwonk\Downloads\개인 프로젝트\juno1412-1\DL\모두의 딥러닝\ml_10_medicalalert_train.csv')
test_df = pd.read_csv(r'C:\Users\kwonk\Downloads\개인 프로젝트\juno1412-1\DL\모두의 딥러닝\ml_10_medicalalert_test.csv')

# 데이터 체크
train_df.head()
test_df.head()

# 3. 데이터 전처리
# string data labeling
train_df = train_df.replace({'gender': {'M':0,'F':1}, 'alert' : {'Yes':0, 'No':1}}).fillna(0.0)

# patientid, timestamp 제거 모델 예측에 필요없음
train_df2 = train_df.drop(columns = ['patientid', 'timestamp'], axis=1)

# object data type -> float변경
# model input
train_df2 = train_df2.apply(pd.to_numeric, errors='coerce').fillna(0.0)
train_df2 = train_df2.values

# 테스트 데이터 변경
# 3. 데이터 전처리
# string data labeling
test_df = test_df.replace({'gender': {'M':0,'F':1}, 'alert' : {'Yes':0, 'No':1}}).fillna(0.0)

# patientid, timestamp 제거 모델 예측에 필요없음
test_df2 = test_df.drop(columns = ['patientid', 'timestamp'], axis=1)

# object data type -> float변경
# model input
test_df2 = test_df2.apply(pd.to_numeric, errors='coerce').fillna(0.0)
# id를 인덱스로 설정
test_df2 = test_df2.set_index('id')
test_df2 = test_df2.values


# 3-2. 데이터 전처리
x = train_df2[:,0:14]


# x[['w', 'pw']] = x[['w', 'pw']].round().astype(int)
# 3-2. 데이터 라벨링
y = train_df2[:,14]
e = LabelEncoder() # 각 class 문자열 -> 숫자로 변환
e.fit(y)
y_label = e.transform(y)


# 모델
# 4. 모델
model = Sequential()
model.add(Dense(32, input_dim=14, activation ='relu'))
model.add(Dense(32, activation ='relu'))
model.add(Dense(16, activation ='relu'))
model.add(Dense(1, activation ='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['acc'])

# adam에서 최고성능 98.88% 정확도 수렴 -> epoch 10
model.fit(x, y_label, epochs=10, batch_size =50, validation_split=0.2)


test_df['pred'] = model.predict(test_df2)
test_df['pred'].to_csv(r'C:\Users\kwonk\Downloads\개인 프로젝트\juno1412-1\DL\모두의 딥러닝\test_result.csv')

print('done')