### input, output

# 로그인 어플리케이션 입력 기능 추가
# in_str = input("아이디를 입력해주세요. \n")
# real_egoing=11
# real_k8805 = "kc"
# if real_egoing == input:
#     print("Hello, egoing")
# elif real_k8805 == input:
#     print("Hello_k8805")
# else:
#     print("who are you!")

# input & output 복습
input_id = input("아이디를 입력하세요.\n")
read_id = "11"
read_pwd = "aaa"
if read_id == input_id:
    print("hi" + " %s" % input_id)
elif read_pwd == input_id:
    print("hi" + "%s" % input_id)
else:
    print("who are you?")