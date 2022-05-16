# 캡슐화
class Unit:
    def __init__(self,name):
        self.__name = name
        self.__hp=100

    def GetHP(self):
        return self.__hp

    def GetName(self):
        return self.__name

    def __SetHP(self, hp):
        if(hp<0):
            hp = 0
        if(hp>100):
            hp = 100
        self.__hp = hp

    def Play(self,hour):
        print(hour,"시간 운동하다.")
        self.__SetHP(self.__hp+hour)

    def Drink(self, cups):
        print(cups,"잔 마시다.")
        self.__SetHP(self.__hp-cups)
        self.


unit = Unit("홍길동")
unit.
print("유닛 이름:{0} 체력:{1}".format(unit.GetName(), unit.GetHP()))