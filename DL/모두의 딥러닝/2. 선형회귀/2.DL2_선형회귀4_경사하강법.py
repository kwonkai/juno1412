# tensorflow 버전1 예시 -> 버전 2 에서 keras.optimizers.SGD()로 변환 가능
# https://www.tensorflow.org/guide/migrate?hl=ko

from random import seed
import numpy as np
import tensorflow as tf

# data list 만들기
# i[0] = list 첫번째 값, i[1] = list 두번째 값
data = [[2, 80], [4, 92], [6, 87], [8, 95]]
x_data = [i[0] for i in data]
y_data = [i[1] for i in data]

learn_rate = 0.1

# 기울기 a, y절편 b 임의값 정하기exi
# Variable() = 변수값 설정
# random.uniform() = 임의의 수를 생성해주는 함수 최소, 최대값만 적어줌
a = tf.Variable(tf.random.uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random.uniform([1], 0, 100, dtype=tf.float64, seed=0))

# 일차방정식 구현
y = a * x_data + b

# 평균제곱근 오차
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# # 경사하강법
gradient_decent = tf.train.GradientDescentOptimizer(learn_rate).minimize(rmse)

# tensorflow를 이용한 학습
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())
    # 2001번 실행(0번 째를 포함하므로)
    for step in range(2001):
        sess.run(gradient_decent)
        # 100번마다 결과 출력
        if step % 100 == 0:
            print("Epoch: %.f, RMSE = %.04f, 기울기 a = %.4f, y 절편 b = %.4f" % (step, sess.run(rmse),sess.run(a),sess.run(b)))

### 다중 선형 회귀 ###
import tensorflow as tf
data = [[2, 0, 2, 81], [4, 4, 2, 93], [6, 2, 4, 91], [8, 3, 4, 97]]

# 3개의 독립변수 만들기
x1 = [x_row1[0] for x_row1 in data]
x2 = [x_row2[1] for x_row2 in data]
x3 = [x_row3[2] for x_row3 in data]
y_data = [y_row[3] for y_row in data]

# 3개의 기울기 구하기
a1 = tf.Variable(tf.random.uniform([1], 0, 10, dtype=tf.float64, seed=0)) # 수업 시간
a2 = tf.Variable(tf.random.uniform([1], 0, 10, dtype=tf.float64, seed=0)) # 과외수업횟수
a3 = tf.Variable(tf.random.uniform([1], 0, 10, dtype=tf.float64, seed=0)) # 자습시간
b = tf.Variable(tf.random.uniform([1], 0, 100, dtype=tf.float64, seed=0))

# 새로운 1차 방정식 만들기
y = a1*x1 + a2*x2 + a3*x3 + b

# 학습률
learn_rate = 0.1

# 오차값
rmse = tf.sqrt(tf.reduce_mean(tf.square(y-y_data)))

# 경사하강법
gradient_decent = tf.train.GradientDescentOptimizer(learn_rate).minimize(rmse)

# 실행세션 만들기

with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.compat.v1.global_variables_initializer())
    # 2001번 실행(0번 째를 포함하므로)
    for step in range(10001):
        sess.run(gradient_decent)
        # 100번마다 결과 출력
        if step % 1000 == 0:
            print("Epoch: %.f, RMSE = %.04f, 기울기 a1 = %.4f, 기울기 a2 = %.4f, 기울기 a3 = %.4f, y 절편 b = %.4f" % (step, sess.run(rmse), sess.run(a1), sess.run(a2), sess.run(a3), sess.run(b)))





# tensorflow 2.0 이상 버전 구문
# import numpy as np
# import pandas as pd
# import matplotlib as plt

# #공부시간 X와 성적 Y의 리스트를 만듭니다.
# data = [[2, 81], [4, 93], [6, 91], [8, 97]]
# x = [i[0] for i in data]
# y = [i[1] for i in data]

# #그래프로 나타내 봅니다.
# plt.figure(figsize=(8,5))
# plt.scatter(x, y)
# plt.show()

# #리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸어 줍니다.(인덱스를 주어 하나씩 불러와 계산이 가능해 지도록 하기 위함입니다.)
# x_data = np.array(x)
# y_data = np.array(y)

# # 기울기 a와 절편 b의 값을 초기화 합니다.
# a = 0
# b = 0

# #학습률을 정합니다.
# lr = 0.03 

# #몇 번 반복될지를 설정합니다.
# epochs = 2001 

# #경사 하강법을 시작합니다.
# for i in range(epochs): # epoch 수 만큼 반복
#     y_hat = a * x_data + b  #y를 구하는 식을 세웁니다
#     error = y_data - y_hat  #오차를 구하는 식입니다.
#     a_diff = -(2/len(x_data)) * sum(x_data * (error)) # 오차함수를 a로 미분한 값입니다. 
#     b_diff = -(2/len(x_data)) * sum(error)  # 오차함수를 b로 미분한 값입니다. 
#     a = a - lr * a_diff  # 학습률을 곱해 기존의 a값을 업데이트합니다.
#     b = b - lr * b_diff  # 학습률을 곱해 기존의 b값을 업데이트합니다.
#     if i % 100 == 0:    # 100번 반복될 때마다 현재의 a값, b값을 출력합니다.
#         print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))

# y_pred = a * x_data + b

# plt.scatter(x, y)
# plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
# plt.show()



