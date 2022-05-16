# ###변수 선언
# name="해리포터" 
# print("안녕하세요."+name+"님") 


# ### 수 계산에서 변수의 사용 
# donation = 200 
# student = 10 
# sponsor = 20 
# print((donation*student)/sponsor)

# ### 컨테이너
# # 문자열 배열
# name = 'kwon'

# # 여러 문자열 묶기 [] = list
# # 변수에 담기
# names = ['kwon', 'kim', 'lee']

# # list에 다양한 형태의 정보가 들어갈 수 있음
# # mixing = 이름, 나이, 직업, 거주지, 자차보유여부
# mixing = ['kim', 20, 'student', 'seoul', True]

# # 값 수정
# mixing[1] = 'busan'
# mixing[2] = mixing[2]*5

# # 값 슬라이싱
# mixing[0:3]



# ### 조건문 제어문
# # if
# money = True
# if money:
#     print("택시를 타세요")


# #if else
# money = 1000
# if money >= 3000:
#     print("버스를 타세요")
# else:
#     print("따릉이를 타세요")


# # if ~ elif ~ else
# pocket = [1000, 3000, 5000]
# if 5000 in pocket:
#     print("택시를 타고 가라")
# elif 3000 in pocket:
#     print("버스를 타라")
# elif 1000 in pocket:
#     print("따릉이를 타라")
# else:
#     print("걸어가자")

# # and / or / not
# money = 2000
# card = True
# if money >= 3000 and card:
#     print("택시를 타고 가라")
# elif money == 2000 or card:
#     print("버스를 타고 가라")
# else:
#     print("걸어가라")


# # in, not in
# pocket = ['cellphone', 'money']
# if 'money' in pocket:
#     print("버스를 타고 가라")
# else:
#     print("따릉이를 타라")

# ### 반복문
# # 3번만 반복하는 반복문 
# i = 0 
# while i < 3:  # i = 0,1,2 #3이면 False로 멈춤 
#     i = i + 1
#     print('Hello world') 
    

# # while, if조건문 break
# i = 0
# while i < 10:
#     if i == 4:
#         break
#     print(i)
#     i = i+1


# # while, if 조건문 continue
# num = 0
# while num < 5:
#     num += 1
#     if num == 3:
#         continue
#     print(num)

# # for 반복문
# members = ['lee', 'kim', 'gong', 'gu']
# for member in members:
#     print(member)
    
# # for문 range 활용하기
# for item in range(5,14):
#     print(item)
    
# # for, if 문
# items = ['a', 'b', 'c', 'd']
# for item in items:
#     if item == 'd':
#         items.append('e')
#     print(items)

# # 중복 for 문
# for i in range(5):
#   for j in range(5):
#   	print(i+j, end=' ')

# ### 함수
# def add(a,b):
#     return a+b

# a=2
# b=1
# add(10,5)

# ### 인수, 파라미터
# def kaka(num1, num2): # parameter = (num1, num2)
#     result = num1 - num2
#     return result
    
# call = kaka(2,1) # argument = (2,1)
# call


# class Class1(object):
#     def method1(self): return 'm1'
# c1 = Class1()
# print(c1, c1.method1())
 
# class Class3(Class1):
#     def method2(self): return 'm2'
# c3 = Class3()
# print(c3, c3.method1())
# print(c3, c3.method2())
 
# # class Class2(object):
# #     def method1(self): return 'm1'
# #     def method2(self): return 'm2'
# # c2 = Class2()
# # print(c2, c2.method1())
# # print(c2, c2.method2())

# ## 캡슐화
# class Cal(object):
#     def __init__(self, v1, v2):
#         self.__v1 = v1 # __ 인스턴스 외부에서 접근하지 못함
#         self.v2 = v2
        
#     def add(self):
#         return self.v1+self.v2
#     def subtract(self):
#         return self.v1-self.v2
        
#     def setV1(self, v):
#         if isinstance(v, int): # int 값만 v 변수로 받을 것
#             self.v1 = v
#     def getV1(self):
#         return self.v1
# c1 = Cal(10,10)
# print(c1.add())
# print(c1.subtract())
# c1.v2 = 30
# print(c1.add())
# print(c1.subtract())


# class C1:
#     def m(self):
#         return 'parent'
# class C2(C1):
#     def m(self):
#         return super().m() + ' child'
#     pass
# o = C2()
# print(o.m())

# def kaka(num1, num2): # parameter = (num1, num2)
#     result = num1 - num2
#     return result
    
# call = kaka(2,1) # argument = (2,1)
# call

# from abc import *
 
# class StudentBase(metaclass=ABCMeta): # 추상메서드
#     @abstractmethod
#     def study(self):
#         pass
 
#     @abstractmethod
#     def go_to_school(self):
#         pass
 
# class Student(StudentBase):
#     def study(self):
#         print('공부하기')
 
#     def go_to_school(self):
#         print('학교가기')
 
# james = Student()
# james.study()
# james.go_to_school()

# items = ['a', 'b', 'c', 'd']
# for item in items:
# 	if item == 'd':
#          items.append('e')
#          print(items)
# for i in range(5):
#     for j in range(5):
#         print(i+j, end="")
#     print()
   

# for i in range(4):
#     for j in range(4):
#         print(i + j, end=" ")
#     print()

from unicodedata import name


class Men:
    def __init__(self, name, number):
        self.__name = name
        self.__number = number

    def get_name(self):
        return self.__name
    
    def set_name(self, name):
        self.__name = name
    
    def get_number(self):
        return self.__number
    
    def set_number(self, number):
        self.__number = number

a = Men("민지", 13)
# name, number, __name, __number 불러와지지 않음
print(a.name) 
print(a.number) 
print(a.__name) 
print(a.__number) 

# get_method로 불러오기
print(a.get_name())
print(a.get_number())


class Men:
    def __init__(self, name, number):
        self.name = name
        self.number = number

    def get_name(self):
        return self.name
    
    def set_name(self, name):
        self.name = name
    
    def get_number(self):
        return self.number
    
    def set_number(self, number):
        self.number = number

a = Men("민지", 13)
# __name, __number 불러와지지 않음
print(a.name) 
print(a.number) 

        