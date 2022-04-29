### Class Member
# class에 소속되어 있는 변수와 메소드를 알아보자
import datetime
d1 = datetime.date(2000, 1, 1)
d2 = datetime.date(2010, 1, 1)

print(d1.year)
print(datetime.date.today())

### Class method

class Cs:
    @staticmethod # 장식자, 꼭 붙여주어야함
    def static_method():
        print("Static method")
    @classmethod # 장식자, 꼭 붙여주어야함
    def class_method(cls): # cls라는 첫번째 매개변수를 가져옴.
        print("Class method")
    def instance_method(self): #
        print("Instance method")



i = Cs()
Cs.static_method()
Cs.class_method()
i.instance_method()


## Class 변수
class Cs:
    count = 0 # class의 안, method의 밖
    def __init__(self): # 초기화
        Cs.count = Cs.count + 1
    @classmethod
    def getCount(cls): # 
        return Cs.count

i1 = Cs()
i2 = Cs()
i3 = Cs()
i4 = Cs()
print(Cs.getCount()) 