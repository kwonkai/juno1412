###변수 선언
name="해리포터" 
print("안녕하세요."+name+"님") 


### 수 계산에서 변수의 사용 
donation = 200 
student = 10 
sponsor = 20 
print((donation*student)/sponsor)

### 컨테이너
# 문자열 배열
name = 'kwon'

# 여러 문자열 묶기 [] = list
# 변수에 담기
names = ['kwon', 'kim', 'lee']

# list에 다양한 형태의 정보가 들어갈 수 있음
# mixing = 이름, 나이, 직업, 거주지, 자차보유여부
mixing = ['kim', 20, 'student', 'seoul', True]

# 값 수정
mixing[1] = 'busan'
mixing[2] = mixing[2]*5

# 값 슬라이싱
mixing[0:3]

### 함수
def add(a, b): 
    return a + b
    
a=2
b=1
add(1,2)


# if ~ elif ~ else
# pockets = [1000, 3000, 5000]
# for porket in pockets:
#    if porket == 5000:
#       print("택시를 타고 가라")
#    elif porket == 3000:
#       print("버스를 타라")
#    elif porket == 1000:
#       print("따릉이를 타라")
#    else:
#       print("걸어가자")

# # in, not in
# pockets = ['cellphone', 'money']
# for pocket in pockets:
#    if 'money' not in pockets:
#       print("버스를 타고 가라")
#    elif 'card' in pockets:
#       print("따릉이를 타라")

# i = 1

# while i < 100:
#    i += 1
#    if i*3 > 50:
#       result = i*3
# print(result)


# books = [ 'a', 'b', 'c', 'd']
# books[0]
# book = [book for book in books if 1 in books[0]]


# and / or / not
# money = 2000
# card = True
# if money >= 3000 and card:
#     print("택시를 타고 가라")
# elif money == 2000 or card:
#     print("버스를 타고 가라")
# else:
#     print("걸어가라")

# # 중복 for 문
# nums = [1, 2, 3]
# plus = [2, 3, 5]
# for num in nums:
# 	for j in plus:
#         	num = num + j
# print(num)

# for i in range(5) : 
#     for j in range(5-i) :
#         print('*', end='')
#     print()

class Class1(object):
    def method1(self): return 'm1'
c1 = Class1()
print(c1, c1.method1())
 
class Class3(Class1):
    def method2(self): return 'm2'
c3 = Class3()
print(c3, c3.method1())
print(c3, c3.method2())
 
# class Class2(object):
#     def method1(self): return 'm1'
#     def method2(self): return 'm2'
# c2 = Class2()
# print(c2, c2.method1())
# print(c2, c2.method2())

## 캡슐화
class Cal(object):
    def __init__(self, v1, v2):
        self.__v1 = v1 # __ 인스턴스 외부에서 접근하지 못함
        self.v2 = v2
        
    def add(self):
        return self.v1+self.v2
    def subtract(self):
        return self.v1-self.v2
        
    def setV1(self, v):
        if isinstance(v, int): # int 값만 v 변수로 받을 것
            self.v1 = v
    def getV1(self):
        return self.v1
c1 = Cal(10,10)
print(c1.add())
print(c1.subtract())
c1.v2 = 30
print(c1.add())
print(c1.subtract())


class C1:
    def m(self):
        return 'parent'
class C2(C1):
    def m(self):
        return super().m() + ' child'
    pass
o = C2()
print(o.m())