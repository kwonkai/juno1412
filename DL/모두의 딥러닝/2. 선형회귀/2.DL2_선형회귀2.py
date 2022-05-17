
# 평균제곱 오차(mean square error)
# 평균제곱근 오차(root mean square error)
import numpy as np

# 기울기 a, y절편 b #임의
ab = [3, 400]

# data list 만들기
# i[0] = list 첫번째 값, i[1] = list 두번째 값
data = [[1, 378], [2, 412], [3, 386], [4, 415]]
x = [i[0] for i in data]
y = [i[1] for i in data]

def predict(x):
    return ab[0]*x + ab[1]

# 평균제곱 MSE -> python함수로
def mse(p,a):
    return ((p-a)**2).mean()

# 평균제곱근 RMSE -> python 함수로
# np.squr() = 제곱근, **2 제곱, mean() 평균값 구하기
def rmse(p, a):
    return np.sqrt(((p -a)**2).mean())

# mse함수에 데이터를 대입해 최종값 구하기
# rmse 함수에 데이터 대입해 최종값 구하기
def mse_val(mse_predict_result, y):
    return mse(np.array(mse_predict_result), np.array(y))

# rmse 함수에 데이터 대입해 최종값 구하기
def rmse_val(rmse_predict_result, y):
    return rmse(np.array(rmse_predict_result), np.array(y))

# 최종값 출력하기
# predict_result 빈 리스트 만들기
mse_predict_result = []
for i in range(len(x)):
    mse_predict_result.append(predict(x[i]))
    print("생산시간=%.f, 실제생산=%.f, 예측생산=%.f" % (x[i], y[i], predict(x[i])))

print("mse최종값 : " + str(mse_val(mse_predict_result,y)))

# predict_result 빈 리스트 만들기
rmse_predict_result = []
for i in range(len(x)):
    rmse_predict_result.append(predict(x[i]))
    print("생산시간=%.f, 실제생산=%.f, 예측생산=%.f" % (x[i], y[i], predict(x[i])))

print("rmse최종값 : " + str(rmse_val(rmse_predict_result,y)))