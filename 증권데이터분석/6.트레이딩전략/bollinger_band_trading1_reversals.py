# 볼린저 밴드 - 네이버 증권 종가

# 모듈경로 지정
import sys
sys.path.append(r'C:\Users\kwonk\Downloads\개인 프로젝트\juno1412-1\증권데이터분석\DB_API')

# 라이브러리 설정
import matplotlib.pyplot as plt
import Analyzer

# NAVER 주가 종가 가져오기
mk = Analyzer.MarketDB()
df = mk.get_daily_price('NAVER', '2022-07-01')

# 볼린저 밴드 구하기
df['MA20'] = df['close'].rolling(window=20).mean() 
df['stddev'] = df['close'].rolling(window=20).std() 
df['upper'] = df['MA20'] + (df['stddev'] * 2)
df['lower'] = df['MA20'] - (df['stddev'] * 2)
df['PB'] = (df['close'] + df['lower']) / (df['upper'] + df['lower'])

# 30일 기준 현금흐름 구하기
# 1. 종가 구하기 : 고가, 저가, 종가의 합을 3으로 나누기
df['II'] = (2*df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']
df['IIP21'] = df['II'].rolling(window=21).sum() / df['volume'].rolling(window=21).sun * 100

df = df.dropna()


# 볼린저 밴드 그래프 그리기
plt.figure(figsize=(9, 9))
plt.subplot(3, 1, 1)
plt.title('Samsung Electronics Bollinger Band(30 day, 2 std)')
plt.plot(df.index, df['close'], color='#0000ff', label='Close')
plt.plot(df.index, df['upper'], 'r--', label = 'Upper band')
plt.plot(df.index, df['MA20'], 'k--', label='Moving average 20')
plt.plot(df.index, df['lower'], 'c--', label = 'Lower band')
plt.fill_between(df.index, df['upper'], df['lower'], color='0.9')


plt.legend(loc='best')
# subplot - 볼린저 밴드 밴드폭 그래프 그리기
plt.subplot(3, 1, 2)
plt.plot(df.index, df['PB'], 'b', label='%B')
plt.grid(True)
plt.legend(loc='best')


plt.subplot(3, 1, 3)
plt.plot(df.index, df['IIP21'], 'g', label='II 21day')
plt.grid(True)
plt.legend(loc='best')
plt.show()



