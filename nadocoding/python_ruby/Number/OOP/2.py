
### 객체와 변수
# 인캡슐레이션(캡슐화)

class C:
    def __init__(self, v):
        self.value = v
    def show(self):
        print(self.value)


c1 = C(10)
print(c1.value)

#value 값 변경
c1.value = 20
print(c1.value)

c1.show()