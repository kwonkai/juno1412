# 주어진 리스트
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