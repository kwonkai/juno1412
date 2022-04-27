### 컨테이너 & 반복문
# Hello world가 계속 반복되고 있다.


members = ['lee', 'kim', 'gong', 'gu']

# 프로그램 중요 규칙1
# 중복되는 걸 제거 한다 = print, member 중복 제거
print(members[0])
print(members[1])
print(members[2])

# 반복문
i = 0
while i < len(members): # len(members) = 4
    print(members[i])
    i = i + 1


# for문 활용하기
members = ['lee', 'kim', 'gong', 'gu']

for member in members:
    print(member)

# for문 range 활용하기
for item in range(5,14):
    print(item)


# 로그인 애플리케이션에서 for문 사용하기
input_id = input("아이디를 입력해주세요. \n")

# real_id= "egoing"
# real_pwd = "8800"
members = ['egoing', '8800']

for id in members:
    if id == input_id:
        print("Hello"+ id)
    import sys
    sys.exit()
print("who are you?")





# if real_id == input:
#     print("Hello, egoing")
# elif real_pwd == input:
#     print("Hello_k8805")
# else:
#     print("who are you!")