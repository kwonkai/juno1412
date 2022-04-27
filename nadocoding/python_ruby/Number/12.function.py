# ### 함수
# # function
# # 일반 코드
# result = len('aaa')
# print(result)

# # 기본 함수와 return 값
# def a3(): # 함수 정의
#     print('aaa') # 함수 본문

# a3() # 함수 호출


# ## 함수의 출력값 return
# # 표현식, Expression
# # return : 함수의 결과값, return을 만나는 즉시 함수를 종료함
# # return을 통한 출력값(return 값)우리가 만든 함수를 더 좋은 부품으로 만들 수 있게끔 함

# def a3():
#     return 'aaa' # return

# a3()

# def a3():
#     a = 'aaa'
#     return a # return
# a3()

# def a3():
#     print('before')
#     return 'aaa' # return : 
#     print('after') # return 다음에 나오는 구문은 실행되지 않음


# # a3 : 범용적으로 사용이 가능
# def a3():
#     return 'aaa' # return


# # a3 : a()함수를 호출하고 'aaa'print해주는 경우에만 호출 가능
# def a3():
#     print('aaa')

# a3()

# def a3_email():
#     email('aaa')

# a3_email()



# ## 함수의 입력값 input
# # 중복의 문제
# # a3, a4 언제나 같은 결과만 출력함
# def a3():
#     print('aaa')

# def a4():
#     print('aaaa')

# print(a4())

# # 해결
# def a(num): # num = 변수값으로 설정
#     return 'a' * num

# print(a(10))



# ## 함수의 입력값 - 여러개의 입력값

# def a(num): # num = 변수값으로 설정
#     return 'a' * num

# print(a(10))

# # 입력값 하나 더 받기
# def make_string(str, num):
#     return str * num

# print(make_string('abc' , 5))


# 로그인 어플리케이션 기능추가
# 함수화
input_id = input("아이디를 입력해주세요. \n")
def login (_id):
    members = ['kim', 'gong', 'lee']
    for member in members:
        if member == _id:
            return True  # member == _id가 일치하면 True를 반환한다.
    return False  # for 문이 끝났을 때, 일치하지 않는다면 False 값을 반환한다.

if login(input_id):
    print('Hello, ' + input_id)
else:
    print('Who are you?')




