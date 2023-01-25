# 4번 퀴즈
dict_first = {'사과':30, '배':15, '감':10, '포도':10}
dict_second = {'사과': 5, '배': 25, '감':15, '포도':25}

def merge_dict(dict_first, dict_second):
    # key 값 사과, 배, 감, 포도 list 만들기
    key_value = ['사과', '배', '감', '포도']

    # 새로운 딕셔너리 생성
    dict_new = {}

    # key 값 반복
    for i in key_value:
        # first, second의 각각 key-value 값 가져오기
        first = dict_first.get(i)
        second = dict_second.get(i)
        # 2개의 dictionary 값 합하기
        total = first + second
        
        # 새로운 값 업데이트
        dict_new.update({i:total})
    
    return dict_new

print(merge_dict(dict_first, dict_second))