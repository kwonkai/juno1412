### 논리연산자
# and / or /not

# 1. and

input_id = input("아이디를 입력해주세요. \n")
input_pwd = input("비밀번호를 입력해주세요 \n")
real_id= "egoing"
real_pwd = "10"

if real_id == input_id:
    if real_id == input_pwd:
        print("Hello")
    else:
        print("Wrong password")
else:
    print('wrong id or password')


# 2. or
in_str = input("아이디를 입력해주세요. \n")
real_egoing= "egoing"
real_k8805 = "k8805"

if real_egoing == input or real_k8805 == in_str:
    print("Hello")
else:
    print("who are you!")