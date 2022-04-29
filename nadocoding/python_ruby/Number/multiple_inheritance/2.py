### 상속의 응용
# 계산기 Cal
# 다중 상속
class CalMultiply(): # Cal이 가진 모든 method를 상속받음
    def multiply(self):
        return self.v1*self.v2

class CalDivide(): # CalMultiply가 가진 모든 method를 상속받음
    def divide(self):
        return self.v1 / self.v2
        
class Cal(CalMultiply, CalDivide):

    def __init__(self, v1, v2): # v1, v2 : __init__내부에서만 사용가능한 지역변수
        if isinstance(v1, int): # isinstance : 매개변수 v1이 int 값인지 알아보는 함수 
            self.v1 = v1
        if isinstance(v2, int):
            self.v2 = v2  
    def add(self):
        return self.v1 + self.v2
    def subtract(self):
        return self.v1 - self.v2
    def setV1(self, v):
        if isinstance(v, int): # v = int 값인지 확인 True라면 실행, False 라면 무시
            self.v1 = v 
    def getV1(self):
        return self.v1

## CalMultiply
class CalMultiply(Cal): # Cal이 가진 모든 method를 상속받음
    def multiply(self):
        return self.v1*self.v2

class CalDivide(CalMultiply): # CalMultiply가 가진 모든 method를 상속받음
    def divide(self):
        return self.v1 / self.v2

c = Cal(100, 10)
print(c.add())
print(c.multiply())
print(c.divide())

c1 = CalMultiply(10,10) # c1은 Cal과 CalMultiply의 method를 모두 사용가능하다.

c2 = CalDivide(30, 10) # c2은 Cal, CalMultiply, CalDivide의 method를 모두 사용가능하다.
