import auth
input_id = input("아이디를 입력하세요. \n")

if auth.login(input_id):
    print('Hello' + input_id)
else:
    print('who are you?')