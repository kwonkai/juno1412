import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.__version__

# data 만들기
data = [[2, 0], [4,0], [6, 0], [8,1], [10,1], [12,1], [14,1]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

# ax+b
# a 기울기값, b = y절편값 설정
# random seed는 random으로 생성한 결과가 항상 같은 value를 갖도록 하는 방법
a = tf.Variable(tf.random.normal([1], dtype=tf.float64, seed=0))
b = tf.Variable(tf.random.normal([1], dtype=tf.float64, seed=0))
# cost_fn = lambda: 5 * a + 3 * b

# 시그모이드 함수 방정식 구현
y = 1/(1+ np.e**(a* x_data + b))

# 오차 = -평균(y*logh + (1-y)log(1-h)) 방정식 구현
# h = y, y_data = y
# treduce_mean() = 평균 구하기
loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1-y))

# 학습률 learning_rate 지정
# 오차 최소값 찾기
learning_rate = 0.5
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# gradient_decent = tf.keras.optimizers.SGD(0.5) # tensorflow v2
# gradient_decent.minimize(cost_fn, [a,b])
# 결과값 출력
# tf.global_variables_initializer() 변수 초기화
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(60001):
        sess.run(gradient_decent)
        if i % 6000 ==0: # i = 6000번이 될 때마다 결과값 print
            print("Epoch: %.f, loss = %.04f, 기울기 a = %.4f, y 절편 b = %.4f" % (i, sess.run(loss),sess.run(a),sess.run(b)))



# 단순 로지스틱 회귀 tensorflow 2.8 버전 시행
# 기울기 a와 절편 b의 값을 초기화 합니다.
# a = 0
# b = 0

# #학습률을 정합니다.
# lr = 0.05 

# #시그모이드 함수를 정의합니다.
# def sigmoid(x):
#     return 1 / (1 + np.e ** (-x))

# #경사 하강법을 실행합니다.
# for i in range(2001):
#     for x_data, y_data in data:
#         a_diff = x_data*(sigmoid(a*x_data + b) - y_data) 
#         b_diff = sigmoid(a*x_data + b) - y_data
#         a = a - lr * a_diff
#         b = b - lr * b_diff
#         if i % 1000 == 0:    # 1000번 반복될 때마다 각 x_data값에 대한 현재의 a값, b값을 출력합니다.
#             print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))
# # 앞서 구한 기울기와 절편을 이용해 그래프를 그려 봅니다.
# x_data = [i[0] for i in data]
# y_data = [i[1] for i in data]

# plt.scatter(x_data, y_data)
# plt.xlim(0, 15)
# plt.ylim(-.1, 1.1)
# x_range = (np.arange(0, 15, 0.1)) #그래프로 나타낼 x값의 범위를 정합니다.
# plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a*x + b) for x in x_range]))
# plt.show()



# 다중 로지스틱 회귀()
# 1. 라이브러리 설정
import tensorflow as tf
import numpy as np

# 2. 실행 시 같은 결과 출력을 위한 seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 3. 데이터 설정하기
x_data = np.array([[2, 0], [4,0], [6, 0], [8,1], [10,1], [12,1], [14,1]])
y_data = np.array([0, 0, 0, 1, 1, 1, 1]).reshape(7,1)

# 4. tf.placeholder로 데이터 담아주기
# tf.placeholder = 입력값을 저장하는 그릇, tf.placeholder(dtype, shape, name)
X = tf.placeholder(tf.float64, shape=[None, 2])
Y = tf.placeholder(tf.float64, shape=[None, 1])

# 5. 기울기a, y절편 b 임의값 설정
# x1(단순) -> x1, x2(다중)로 변수 추가
# 변수 증가 -> 기울기 각각 계산
# ax -> a1x1 + a2x2 

a = tf.Variable(tf.random.uniform([2,1], dtype=tf.float64)) #[2,1] 의미 : 들어오는 값은 2개 나가는 값은 1개 (변수가 2개) 
b = tf.Variable(tf.random.uniform([1], dtype=tf.float64))

# 6. sigmoid 함수 사용
# tensorflow는 matmul()을 이용해 행렬곱 적용
# ???
y = tf.sigmoid(tf.matmul(X, a) +b) 

# 7. 오차 구하기
# 오차 = -평균(y*logh + (1-y)log(1-h)) 방정식 구현
loss = -tf.reduce_mean(y * tf.log(y) + (1-Y)*tf.log(1-y))

# 8. 학습률 설정
learning_rate = 0.05

# 9. 오차 최소값 찾기
# tf.cast() 조건에 따른 True, False 판단 기준에 따라 True=1, False=0으로 반환한다.
# tf.equal() 두 값이 동일하면 True, 다르면 False로 반환하는 함수이다.
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


predicted = tf.cast(y > 0.5, dtype=tf.float64) # y값이 0.5이상이라면 True로 반환한다.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64)) # predicted와 Y값이 같다면 True=1로 반환한다. predicted와 Y값이 다르면 False=0로 반환한다.

# 10. 학습 및 결론 도출
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(3001): # 잘모르겠다 ㅠㅠ
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={X: x_data, Y: y_data})
        if(i + 1) % 300 == 0:
            print("step=%d, a1=%.4f, a2=%.4f, b=%.4f, loss=%.4f" % (i+1, a_[0], a_[1], b_, loss_))

# 11. 실제 데이터 적용하기
# (7,6) = 7,6은 공부한 시간과 과외 수업 횟수
# feed_dict = placeholder에 new_x 데이터를 하나씩 읽고 실행함.
    new_x = np.array([7,6.]).reshape(1,2)
    new_y = sess.run(y, feed_dict={X: new_x})

    print("공부한시간: %d, 과외 수업 횟수: %d" % (new_x[:,0], new_x[:,1]))
    print("합격 가능성 : %6.2f %%" % (new_y*100))

