### 문자와 데이터
from winreg import HKEY_LOCAL_MACHINE


print('Hello')
print("Hello")
print("'Hello world'")
print("Hello 'world'")
print('Hello "world"')

### 문자의 연산(=제어법)
print('Hello ')
print('world')
print('Hello' + 'world') # =print('Hello world')

print('Hello'*5) # 곱한 숫자만큼 반복
print('Hello'[0])
print('Hello'[2])
print('Hello'[4])

### 문자의 연산2
# capitalize() 첫글자를 대문자로
print('hello world'.capitalize())

# 모든글자를 대문자로
print('hello world'.upper())

#글자수세기
print('hello world'.__len__())
print(len('hello world'))

#글자 바꾸기
print('hello world'.replace('world', 'programming'))


### 특수문자
# \(역슬래쉬) : escape
print("egoing's \"tutorial\"")
print("\\") # \ or \\\는 에러(닫히지 않은 문자열)
print("hello \n world") # \n(=new lins) = 줄바꿈
print("hello \t\t world") # \t(=tab) = 들여쓰기 \t\t 들여쓰기 2번
print("\a") #컴퓨터 기본 경고음
print("Hello \n world")


### 데이터 타입
print(10+5) # int 10 + 5
print("10"+"5") # string 10 + 5