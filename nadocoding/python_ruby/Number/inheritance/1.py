### 상속 inheritance
# 상속의 문법

class Class1(object):
    def method1(self):
        return 'm1'

c1 = Class1()
print(c1.method1())

## 상속이 없을 때 
class Class2(object):
    def method1(self):
        return 'm1'
    
    def method2(seld):
        return 'm2'

c2 = Class2()
print(c2.method1())
print(c2.method2())

## 상속이 있을 때
class Class3(Class1): # class3 = class1의 기능을 가지고 method2의 기능을 추가함
    def method2(self):
        return 'm2'

c3 = Class3()
print(c3, c3.method1())
print(c3, c3.method2())



### 상속의 응용
# 계산기 Cal
class Cal(object):

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


c1 = CalMultiply(10,10) # c1은 Cal과 CalMultiply의 method를 모두 사용가능하다.
print(c1.add())
print(c1.multiply())

c2 = CalDivide(30, 10) # c2은 Cal, CalMultiply, CalDivide의 method를 모두 사용가능하다.
print(c2, c2.add())
print(c2, c2.multiply())
print(c2, c2.divide())