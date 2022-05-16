
### 객체와 변수
# 클래스 생성자

class Cal(object):
    def __init__(self, v1, v2): # 생성자
        print(v1, v2)
 
c1 = Cal(10,10)

### 인스턴스 변수 & 메소드
class Cal(object):
    def __init__(self, v1, v2): # 생성자 # self : 인스턴스 변수
        self.v1 = v1
        self.v2 = v2

    def add(self): # 인스턴스 메소드 # self : 인스턴스 변수
        return self.v1 + self.v2  # self.v1 : 인스턴스 변수 v1/v2 사용가능

    def substract(self): # 인스턴스 메소드 # self : 인스턴스 변수
        return self.v1 - self.v2  # self.v1 : 인스턴스 변수 v1/v2 사용가능
 
c1 = Cal(10,10)
print(c1.add())

c2 = Cal(40,50)
print(c1.substract())