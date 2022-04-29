### Override
# 재정의

class C1: # class
    def m(self): # method
        return 'parent'

class C2(C1): # Class C1을 상속받음
    def m(self):
        # super().m # 부모 class의 method를 사용한다.
        return  super().m() + ' children'
    pass # pass = method가 존재하지 않는 클래스 pass

k = C2()
print(k.m())
