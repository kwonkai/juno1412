
### 객체 지향 프로그래밍의 사례
name = 1 # 'kim' = 객체
names = [1, 2, 3, 4] # list 데이터 타입 -> 배열이면서 객체 타입

## 객체 사용
# 계산기 만들기
# 클래스 선언하기
class Cal(object):

# 생성자(constructor)
# python 의 method들은 첫번째 매개변수(=첫번째 instance)를 꼭 정의해야 한다. 첫번째 인스턴스 = 첫번째 매개변수
    def __init__(self, v1, v2): # v1, v2 : __init__내부에서만 사용가능한 지역변수
        print(v1, v2)
        self.v1 = v1
        self.v2 = v2  
    def add(self):
        return self.v1 + self.v2
    def subtract(self):
        return self.v1 - self.v2

# 인스턴스 변수와 메소드
c1 = Cal(10,10)
print(c1.add())
print(c1.subtract())


c2 = Cal(30,20)
print(c1.add())
print(c1.subtract())