### modules - 로그인 애플리케이션 적용
def login(_id, _pwd):
    members = ['kim', 'gong', 'lee']
    passwords = ['1111','2222','3333']
    for member in members:
        if member == _id:
            if _pwd in passwords:       
                return True
            else:
                return False
    return False

### modules - 로그인 애플리케이션 적용
# def login1(_id, _pwd):
#     members = ['kim', 'gong', 'lee']
#     passwords = [1111,2222,3333]
#     for member in members:
#         for password in passwords:
#             if member == _id and password == _pwd:    
#                 return True
#     return False
# def login(_id):
#     members = ['egoing', 'k8805', 'leezche']
#     for member in members:
#         if member == _id:
#             return True
#     return False