# 다중 로지스틱 회귀()
# 1. 라이브러리 설정
import tensorflow as tf
import numpy as np

# 2. 실행 시 같은 결과 출력을 위한 seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 3. 데이터 설정하기
x_data = np.array([[2, 5], [4,6], [6, 2], [8,4], [10,7], [12,9], [14,8]])
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
y = tf.sigmoid(tf.matmul(X, a) +b) 

# 7. 오차 구하기
# 오차 = -평균(y*logh + (1-y)log(1-h)) 방정식 구현
loss = -tf.reduce_mean(Y * tf.log(y) + (1-Y)*tf.log(1-y))

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
    
    for i in range(3001):
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={X: x_data, Y: y_data}) # feed_dict():placeholder에 저장된 데이터 [x1, x2를 하나씩] 실행한다.
        if(i + 1) % 300 == 0:
            print("step=%d, a1=%.4f, a2=%.4f, b=%.4f, loss=%.4f" % (i + 1, a_[0], a_[1], b_, loss_))


# 11. 실제 데이터 적용하기
# (7,6) = 7,6은 공부한 시간과 과외 수업 횟수
# feed_dict = placeholder에 new_x 데이터를 하나씩 읽고 실행함.
    x1 = np.array([5,6.]).reshape(1,2)
    y1 = sess.run(y, feed_dict={X: x1})

    print("공부 시간: %d, 개인 과외 수: %d" % (x1[:,0], x1[:,1]))
    print("합격 가능성: %6.2f %%" % (y1*100))

