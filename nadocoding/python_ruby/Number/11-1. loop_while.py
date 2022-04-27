### 반복문
# 반복문이 없는 경우
print("Hello world 0")
print("Hello world 9")
print("Hello world 18")
print("Hello world 27")
print("Hello world 36")
print("Hello world 45")
print("Hello world 54")
print("Hello world 63")
print("Hello world 72")
print("Hello world 81")

# while 반복문
# while True:
#     print('Hello world')
# while False:
#     print('Hello world')
# print('After while')


# 반복문 없는 중복
print("Hello world")
print("Hello world")
print("Hello world")

# 3번만 반복하는 반복문
i = 0
while i < 3: # i = 0,1,2 #3이면 False로 멈춤
    print('Hello world')
    i = i + 1


# 반복문의 활용
i = 0
while i < 10: 
    print('print("Hello world '+ str(i*9)+'")')
    i = i + 1


# if조건문 while 반복문 합해서 사용할 경우 사용할 경우
i = 0
while i < 10:
    if i == 4:
        break # i값이 4라면 반복을 멈춤 = break
    print(i)
    i = i + 1
print("after while")


# except
# for문으로 반복할 경우

for i in range(10):
    print('print("Hello world' + str(i*7)+'")')
    i = i + 1


