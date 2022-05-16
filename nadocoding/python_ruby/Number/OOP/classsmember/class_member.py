class Cs:
    @staticmethod # 정적 메소드
    def static_method():
        print("static method")

    @classmethod
    def class_method(cls):
        print("class method")

    def instance_method(self):
        print("instance method")


i = Cs()
Cs.static_method()
Cs.class_method()
i.instance_method()


class Cs2:
    count = 0 # Class 변수 # 클래스의 안, method의 밖에 변수 선언 시, classs 변수가 됨
    def __init__(self): # 생성자
        Cs.count = Cs.count + 1

    @classmethod # instance 소속에서 class 소속으로 설정
    def getcount(cls):
        return Cs.count

i1 = Cs()
i2 = Cs()
i3 = Cs()
i4 = Cs()
print(Cs.getCount())
