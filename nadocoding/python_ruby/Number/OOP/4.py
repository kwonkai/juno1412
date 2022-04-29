
### 객체와 변수
# 인캡슐레이션(캡슐화)
# set method & get method

class C:
    def __init__(self, v):
        self.__value = v  # '__'를 변수 앞에 붙이게되면 instance 밖에서 접근할 수 없는 상태가 된다.
    def show(self):
        print(self.__value)
    # def getValue(self): # value 의 값을 가져온다
    #     return self.value
    # def setValue(self, v): # setvalue = v
    #     self.value = v


c1 = C(10)

# print(c1.__value())
c1.show()


