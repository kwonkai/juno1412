### modules - 로그인 애플리케이션 적용
def login(_id):
    members = ['kim', 'gong', 'lee']
    for member in members:
        if member == _id:
            return True
    return False

