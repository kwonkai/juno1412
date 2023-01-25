# Quits 1
from re import A
from tkinter import N

class Score():
    def __init__(self, mid, final):
        self.mid = mid
        self.final = final

    def score_int(self, mid, final):
        self.mid = mid
        self.final = final
        return (mid+final) / 2
         

# 출력함수
score = Score(50, 75)
print((score.mid + score.final) / 2)


# Quits 2
# car class 생성
# fuel, wheel 연료/휠 생성
class Car():
    def __init__(self, fuel, wheels):
        self.fuel = fuel
        self.wheels = wheels



class Bike(Car):
    # bike(자식 클래스)에서 size parameter 생성
    def __init__(self, fuel, wheels, size):
        super().__init__(fuel, wheels)
        self.size = size
        

# 출력 예시
bike = Bike("gas", 2, "small")
print (bike.fuel, bike.wheels, bike.size) 


# Quits 3
# 라이브러리 설치 pandas
# terminal 에서 pip install pandas or conda install pandas
import pandas as pd

file_path = 'C:/Users/kwonk/Downloads/개인 프로젝트/juno1412-1/naver_study/data_files/data-01-test-score.csv'

def read_file(file_path):
    # TODO
    # 1. 파일 불러오기
    df = pd.read_csv(file_path, header=None)

    # 2. 빈 리스트 생성
    df_list = []

    # 3. index 숫자만큼 반복문으로 list 변환하여 리스트에 추가
    for i in range(len(df.index)):
        df_list.append(list(df.loc[i].values))

    return df_list


df_list = read_file(file_path)
print(df_list)


# Quits 4
# 라이브러리 설치 pandas
# terminal 에서 pip install pandas or conda install pandas
import pandas as pd

file_path = 'C:/Users/kwonk/Downloads/개인 프로젝트/juno1412-1/naver_study/data_files/data-01-test-score.csv'

def read_file(file_path):
    # TODO
    # 1. 파일 불러오기
    df = pd.read_csv(file_path, header=None)

    # 2. 빈 리스트 생성
    df_list = []

    # 3. index 숫자만큼 반복문으로 list 변환하여 리스트에 추가
    for i in range(len(df.index)):
        df_list.append(list(df.loc[i].values))

    return df_list

def merge_list():

    # 4. 빈 리스트 생성
    df_all = []

    # 5-2. 
    for i in range(len(df_list)):
        for j in range(len(df_list[i])):
            df_all.append(df_list[i][j])

    return df_all

# df, df_list 전역변수 설정
df_list = read_file(file_path)
print(df_list)

df_all = merge_list()
print(df_all)


# Quits 5
# 라이브러리 설치 pandas
# terminal 에서 pip install pandas or conda install pandas
import pandas as pd

file_path = 'C:/Users/kwonk/Downloads/개인 프로젝트/juno1412-1/naver_study/data_files/data-01-test-score.csv'

def read_file(file_path):
    # TODO
    # 1. 파일 불러오기
    df = pd.read_csv(file_path, header=None)

    # 2. 빈 리스트 생성
    df_list = []

    # 3. index 숫자만큼 반복문으로 list 변환하여 리스트에 추가
    for i in range(len(df.index)):
        df_list.append(list(df.loc[i].values))

    return df_list

def merge_list():

    # 4. 빈 리스트 생성
    df_mean = []

    # 5. df_list 길이만큼 반복하며 평균 구해주기
    # sum(list(map(int, df_list[i]))) : list 내부의 list를 뽑아오면서 int값 변경 후 다시 list로 변경

    for i in range(len(df_list)):
        df_mean.append(sum(list(map(int, df_list[i]))) / len(df_list[i]))

    # 평균 오름차순 정리
    df_mean.sort()
    return df_mean

# df, df_list 전역변수 설정
df_list = read_file(file_path)
print(df_list)

df_mean = merge_list()
print(df_mean)
