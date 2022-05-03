### 집합 자료형
# 집합 자료형 만들기

a = set([1,3,5])
a

b = set("List")
b

# set 자료형 indexing
l1 = list(a)
l1

t1 = tuple(a)
t1

## 교집합, 합지합, 차집합 구하기
# set 자료형 만들기
s1 = set([1, 2, 3, 4, 5])
s2 = set([3, 4, 5, 6, 7])

## 1. 교집합
# & : 교집합 기호
s1 & s2

# intersection : 교집합 함수
s1.intersection(s2)
s2.intersection(s1)


## 2. 합집합
# | : 합집합 기호
s1 | s2

# union() : 합집함 함수
s1.union(s2)
s2.union(s1)


## 3. 차집합
# - : 차집합 기호
s1 - s2
s2 - s1

# difference() : 차집합 함수
s1.difference(s2)
s2.difference(s1)


## 깂 1개 추가하기
s1 = set([1, 2, 3, 4, 5])
s1.add(16)
s1

## 값 여러개 추가하기
s1 = set([1, 2, 3, 4, 5])
s1. update([15, 46, 94])
s1

## 특정값 제거하기
s1 = set([1, 2, 3, 4, 5])
s1.remove(4)

s1
