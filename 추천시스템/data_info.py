
import random
import pandas as pd
# 빈 DataFrame 생성하기
df = pd.DataFrame(columns=['userid', 'score'])
for idx in range(1, 30000):
       idx = random.randint(1, 5)
       number = random.randint(1, 5)
       # DataFrame에 특정 정보를 이용하여 data 채우기
       df = df.append(pd.DataFrame([[idx, number]], columns=['userid', 'score']), ignore_index=True)
df.to_csv('score1.csv')
df = df

movie = pd.read_csv('firstkino-data/data-files/movie_list_result.csv')
movie = movie['movieCd']

movie_score = pd.concat([movie, df], axis=1, ignore_index=False)
movie_score.set_index('movieCd', inplace=True)
movie_score.to_csv('score2.csv', encoding='utf-8')