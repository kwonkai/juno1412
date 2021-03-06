from abc import *

class AbstractCountry(metaclass=ABCMeta):
    name = '국가명'
    population = '인구'
    capital = '수도'


    def show(self):
        print("국가 클래스의 method 입니다.")

    # abstract method 추가하기
    @abstractmethod
    def show_capital(self):
        print("국가의 수도는?")

class Korea(AbstractCountry):
    def __init__(self, name, population, captial):
        self.name = name
        self.population = population
        self.capital = captial

    def show_name(self):
        print("국가 이름은 : " + self.name)

    def show_capital(self):
        super().show_capital()
        print("국가 수도 : " + self.capital)



a = AbstractCountry()
a.show()

k = Korea("대한민국", 50000000, '서울')
k.show_name()
k.show_capital()
        
