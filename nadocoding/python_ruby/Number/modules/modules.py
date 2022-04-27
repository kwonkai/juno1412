
## 함수의 도입
# 함수가 중복되어 덮어쓰기가 발생하는 문제
from egoing import a as z # modules_egoing라는 모듈로부터 a라는 함수를 import 한다.
import k8800 as k # modules_egoing, k8800이라는 py 파일을 찾음

print(z())
print(k.a())