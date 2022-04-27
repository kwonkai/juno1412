### 논리연산자
# and / or /not

# 1. and
# 1-1. and로 통합해 id=pwd 맞춘 경우

input_id = input("아이디를 입력해주세요. \n")
input_pwd = input("비밀번호를 입력해주세요 \n")
real_id= "egoing"
real_pwd = "10"

# 실제 id = 입력 id 그리고 실제 비밀번호 = 입력 비밀번호여야 한다.
# id, pwd가 일치하면 Hello, 불일치하면 wrong password
if real_id == input_id and real_pwd == input_pwd:
    print("Hello")
else:
    print("Wrong password")

# 1-2. if 중첩문으로 id=pwd 맞춘 경우
input_id = input("아이디를 입력해주세요. \n")
input_pwd = input("비밀번호를 입력해주세요 \n")
real_id= "egoing"
real_pwd = "10"

# read_id = input_id가 일치한다면
# read_pwd = input_pwd가 일치할 경우 "Hello!"
# pwd 불일치라면 wrong password!
# id 불일치라면 wrong id!

if real_id == input_id:
    if real_pwd == input_pwd:
        print("Hello!")
    else:
        print("wrong password!")
else:
    print("wrong id!")




# 2. or
in_str = input("아이디를 입력해주세요. \n")
real_egoing= "egoing"
real_k8805 = "k8805"

if real_egoing == input or real_k8805 == in_str:
    print("Hello")
else:
    print("who are you!")
