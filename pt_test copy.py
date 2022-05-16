# class Cal(object):
#     def __init__(self, v1, v2):
#         self.__v1 = v1 # __ 인스턴스 외부에서 접근하지 못함
#         self.__v2 = v2
    
#     def setV1(self, v1):
#         if isinstance(v1, int): # int 값만 v 변수로 받을 것
#             self.__v1 = v1
#     def setV2(self ,v2):
#         if isinstance(v2, int): # int 값만 v 변수로 받을 것
#             self.__v2 = v2

#     def getV1(self):
#         return self.__v1
#     def getV1(self):
#         return self.__v2
        
#     def add(self):
#         return self.__v1+self.__v2
#     def subtract(self):
#         return self.__v1-self.__v2
        
# c1 = Cal(10,10)
# print(c1.add())
# print(c1.subtract())

# c1.v2 = 30
# print(c1.add())
# print(c1.subtract())

# ##
# class Cal(object):
#     def __init__(self, v1, v2):
#         if isinstance(v1, int):
#             self.v1 = v1
#         if isinstance(v2, int):
#             self.v2 = v2
#     def add(self):
#         return self.v1+self.v2
#     def subtract(self):
#         return self.v1-self.v2
#     def setV1(self, v):
#         if isinstance(v, int):
#             self.v1 = v
#     def getV1(self):
#         return self.v1
# c1 = Cal(10,10)
# c1.
# print(c1.add())
# print(c1.subtract())
# c1.setV1('one')
# c1.v2 = 30
# print(c1.add())
# print(c1.subtract())




# #접근 지정

# class Unit:
#     def __init__(self,name):
#         self.__name = name
#         self.__hp=100

#     def GetHP(self):
#         return self.__hp

#     def GetName(self):
#         return self.__name

#     def __SetHP(self, hp):
#         if(hp<0):
#             hp = 0
#         if(hp>100):
#             hp = 100
#         self.__hp = hp

#     def Play(self,hour):
#         print(hour,"시간 운동하다.")
#         self.__SetHP(self.__hp+hour)

#     def Drink(self, cups):
#         print(cups,"잔 마시다.")
#         self.__SetHP(self.__hp-cups)


# unit = Unit("홍길동")
# unit.
# print("유닛 이름:{0} 체력:{1}".format(unit.GetName(), unit.GetHP()))
# # unit.

class C1:
    def m(self):
        return 'parent'
class C2(C1):
    def m(self):
        return super().m() + ' child'
o = C2()
print(o.m())