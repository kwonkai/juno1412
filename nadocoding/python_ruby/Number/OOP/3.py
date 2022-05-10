
### 객체와 변수
# 인캡슐레이션(캡슐화)
# set method & get method

class C:
    def __init__(self, v):
        self.value = v
    def show(self):
        print(self.value)
    def getValue(self): # value 의 값을 가져온다
        return self.value
    def setValue(self, v): # setvalue = v
        self.value = v


c1 = C(10)
print(c1.getValue())
c1.setValue(20)
print(c1.getValue())

# c1.getValue()
# c1.setValue()
# c1.show()


## set&get method

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

    def setV1(self, v): # instance 변수의 값을 설정
        if isinstance(v, int): # v = int 값인지 확인 True라면 실행, False 라면 무시
            self.v1 = v 

    def getV1(self): # instance 변수의 값 가져오기
        return self.v1

c1= Cal(10,10)
print(c1.add())
print(c1.subtract())


c1.setV1('one')
c1.v2 = 30
print(c1.add())
print(c1.subtract())


