# 계산기 Cal
class Cal(object):
    _history = [] # _history 정의
    def __init__(self, v1, v2): # v1, v2 : __init__내부에서만 사용가능한 지역변수
        if isinstance(v1, int):
            self.v1 = v1
        if isinstance(v2, int):
            self.v2 = v2  

    def add(self):
        result = self.v1 + self.v2 # result 변수 정의
        Cal._history.append("add : %d+%d=%d" % (self.v1, self.v2, result) ) # Cal class의 _history 변수에 추가함
        return result

    def subtract(self):
        result = self.v1 - self.v2
        Cal._history.append("subtract : %d+%d=%d" % (self.v1, self.v2, result))
        return result

    def setV1(self, v):
        if isinstance(v, int): 
            self.v1 = v 

    def getV1(self):
        return self.v1


    @classmethod # classmethod
    def history(cls): # 첫번째 인자는 cls로 정의
        for item in Cal._history: # _ = 내부에서만 사용할 변수
            print(item)
    def info(self): # Cal의 info method 설정 = Cal의 v1 : self.v1, v2 : self.v2
        return "Cal => v1 : %d, v2 : %d" % (self.v1, self.v2)

## CalMultiply
class CalMultiply(Cal): # Cal이 가진 모든 method를 상속받음
    def multiply(self):
        return self.v1*self.v2
    def info(self): 
        return "CalMultiply => %s" % super().info() # info method가 Cal의 classmethod info()를 가져옴 # %s % = str 값을 나타냄 
class CalDivide(CalMultiply): # CalMultiply가 가진 모든 method를 상속받음
    def divide(self):
        return self.v1 / self.v2
    def info(self): # info method가 # info method가 CalMultiply의 classmethod info()
        return "CalDivide => %s " % super().info()

c0 = Cal(30, 60)
print(c0.info())

c1 = CalMultiply(30,10) # c1은 Cal과 CalMultiply의 method를 모두 사용가능하다.
print(c1.info())

c2 = CalDivide(20, 10) # c2은 Cal, CalMultiply, CalDivide의 method를 모두 사용가능하다.
print(c2.info())