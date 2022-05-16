class Cal(object):
    _history = [] #클래스 변수 설정 # class 내부 공통 변수

    def __init__(self, v1, v2): # 생성자 # 인스턴스 매서드
        if isinstance(v1, int):
            self.v1 = v1
        if isinstance(v2, int):
            self.v2 = v2

    def add(self):
        result =  self.v1 + self.v2
        Cal._history.append("add : %d + %d = %d" % (self.v1, self.v2, result))
        return result

    def substract(self):
        result = self.v1 - self.v2
        Cal._history.append("substract : %d - %d = %d" % (self.v1, self.v2, result))
        return result
    def setV1(self, v):
        if isinstance(v, int):
            self.v1 = v
    def getV1(self):
        return self.v1
    
    @classmethod
    def history(cls):
        for item in Cal._history: #_history : 내부에서만 사용할 변수
            return print(item)



class CalMultiply(Cal):
    def multiply(self):
        result = self.v1*self.v2
        return result

class CalDivide(CalMultiply):
    def divide(self):
        result = self.v1/self.v2
        return result

c1 = CalMultiply(20,10)
print(c1.add())
Cal.history()
print(c1.substract())
Cal.history()
# c2 = CalDivide(20,10)
# # print(c2, c2.add())
# # print(c2, c2.multiply())
# # print(c2, c2.divide())
