## 모듈 modules

# 내장모듈
import math  # math라는 이름의 모듈을 가져온다.
math.ceil(2.6) # 올림 : 2.6보다 큰 정수를 가져옴
math.floor(2.7) # 내림 : 2.7보다 작은 정수를 가져옴
math.sqrt(16) # 루트 : 16의 루트 값을 가져옴


## 모듈이 없을 때

def kim_a():
    return 'a'

# 매우 많은 코드
#~~~
# 매우 많은 코드

def lala_a():
    return 'B'  # 뒤에 정의된 a()값으로 정의되어 a()의 결과값이 바뀜
                # 프로젝트 복잡도 상승 -> solution?? 정리정돈/이름변형 -> 중복/덮어쓰기 문제 해결

# 매우 많은 코드
print(lala_a())



# ## 함수의 도입
# # 함수가 중복되어 덮어쓰기가 발생하는 문제
# from egoing import a as z # modules_egoing라는 모듈로부터 a라는 함수를 import 한다.
# import k8800 as k # modules_egoing, k8800이라는 py 파일을 찾음

# print(z())
# print(k.a())