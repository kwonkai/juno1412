
# 평균제곱근 오차(root mean square error)
import numpy as np

# 기울기 a, y절편 b
ab = [3, 76]

# data list 만들기
# i[0] = list 첫번째 값, i[1] = list 두번째 값
data = [[2, 80], [4, 92], [6, 87], [8, 95]]
x = [i[0] for i in data]
y = [i[1] for i in data]

def predict(x):
    return ab[0]*x + ab[1]

# 평균제곱근 공식 -> python 함수로
# np.squr() = 제곱근, **2 제곱, mean() 평균값 구하기
def rmse(p, a):
    return np.sqrt(((p -a)**2).mean())

# rmse 함수에 데이터 대입해 최종값 구하기
def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))

# 최종값 출력하기
predict_result =[]
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부시간=%.f, 실제점수=%.f, 예측점수=%.f" % (x[i], y[i], predict(x[i])))