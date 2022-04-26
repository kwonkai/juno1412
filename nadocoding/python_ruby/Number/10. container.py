### 컨테이너
# python container = list

# 문자열
'lalala'

# type
type('lalala')
print(type('lalala')) # type('lalala') 불러오기


name = 'kwon'
print(name)

# 여러 문자열 묶기 [] = list
['kwon', 'kim', 'lee']
print(type(['kwon', 'kim', 'lee']))

# 변수에 담기
names = ['kwon', 'kim', 'lee']
print(names)


### 
# 인덱스 index(=색인)
# ['kwon', 'kim', 'lee'] -> index [0, 1, 2]
print([names[1]]) # 'kim'


# list에 다양한 형태의 정보가 들어갈 수 있음
# mixing = 이름, 나이, 직업, 거주지, 자차보유여부
mixing = ['kim', 20, 'student', 'seoul', True]

# 값 변경
mixing[1] = 'busan'
print(mixing)


