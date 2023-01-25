# 1번 퀴즈
num_list = [1, 5, 7, 15, 16, 22, 28, 29]

def get_odd_num(num_list):
    #TODO
    odd_num =[]
    for i in range(len(num_list)):
        if num_list[i] % 2 == 1:
            odd_num.append(num_list[i])
            
    return odd_num

print(get_odd_num(num_list))

# 2번 퀴즈
sentence = "way a is there will a is there where"

def reverse_sentence(sentence):
    # TODO
    # 빈 문자열string 생성
    result = ""
    # 단어 단위로 나눠주기
    st = list(sentence.split(' '))

    # 반복문으로 단어를 문장으로 합쳐주기
    for i in st:
        result += (i+' ')

    return result

print(reverse_sentence(sentence))

# 3번 퀴즈
score = [(100,100), (95, 90), (55, 60), (75, 80), (70, 70)]

def get_avg(score):
    # score list 내부 데이터 개수 i만큼 반복
    for i in range(len(score)):
        # 빈 list 생성
        score_sum=[]
        # 내부 튜플의 데이터 개수 j만큼 반복
        # 빈 list = score_sum에 추가
        for j in range(len(score[i])):
            score_sum.append(score[i][j])
        
        # i번 평균, 출력
        print(i+1,"번, 평균 :",  sum(score_sum) / len(score_sum))
    return

print(get_avg(score))

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

# 5번 퀴즈
# 라이브러리 re 추가
# re.sub(pattern, replacement, string)
# string에서 정규표현식의 pattern과 일치하는 내용을 replacement로 변경
# 만약 빈 문자열("")로 변경하면 패턴에 해당하는 문제만 제거

import re
# 주어진 string
inputs = "cat32dog16cow5"

def find_string(inputs):
    # re.sub에서 0-9 숫자를 제거 후 리턴
    inputs_2 = re.sub(r"[0-9]","", inputs)
    return inputs_2

string_list = find_string(inputs)
print(string_list)   
