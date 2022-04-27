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


### 컨테이너 list 심화
names = ['kim', 'lee', 'jong', 'in']
'kim' in names # True
'jun' in names # False

# len, min, max
num = [ 1, 5, 10, 35, 54]
len(num)
min(num)
max(num)

# append, revers
num.append(68)
print(num)

num.reverse()
print(num)

del(num[0])
print(num)

## 예제 풀어보기
# 2. 여러 list
lists = [[]] * 5
lists

lists[0].append(5)
lists

# 반복문
list = [ [] for i in range(3)]
list[0].append(3)
list[1].append(5)
list[2].append(7)

list

