
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
