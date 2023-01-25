# car class 생성
# fuel, wheel 연료/휠 생성
class Car():
    def __init__(self, fuel, wheels):
        self.fuel = fuel
        self.wheels = wheels



class Bike(Car):
    # bike(자식 클래스)에서 size parameter 생성
    def __init__(self, fuel, wheels, size):
        super().__init__(fuel, wheels)
        self.size = size
        

# 출력 예시
bike = Bike("gas", 2, "small")
print (bike.fuel, bike.wheels, bike.size) 