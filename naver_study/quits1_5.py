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
