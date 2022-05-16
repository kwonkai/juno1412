## 로그인 기능 함수
#input
input_id = input("아이디를 입력해주세요.\n")

# login 함수
def login(_id):
    members = ['egoing', 'k8805', 'leezche']
    for member in members:
        if member == _id:
            return True
    return False
    
# id가 일치한다면 hello!+id , 일치하지 않는다면 who are you?
if login(input_id):
    print('Hello, '+input_id)
else:
    print('Who are you?')