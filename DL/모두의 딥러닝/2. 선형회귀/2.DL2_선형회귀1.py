# 최소제곱법(method of least squares)
import numpy as np

# X = 공부한시간, Y = 성적
# list 형식 정의
x = [2,4,6,8] 
y = [81, 93, 91, 97]

mx = np.mean(x)
my = np.mean(y)


# 분모 
# (x 각 원소값 - x평균값) 제곱
divisor = sum([(mx-i)**2 for i in x])

# 분자
# (x 각 원소값 - x평균값)*(y 각 원소값 - y 평균값) 더하기
def top(x, mx, y, my):
    d=0
    for i in range(len(x)): # x개수만큼 실행
        d += (x[i] - mx) * (y[i] - my)
    return d
    
dividend = top(x, mx, y, my)

# 기울기 a
a = dividend / divisor

# y절편 b
b = my -(mx*a)

# 출력하기
print("기울기 a =", a)
print("y 절편 b", b)