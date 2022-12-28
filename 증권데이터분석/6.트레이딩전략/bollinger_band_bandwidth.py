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

# 볼린저밴드 밴드폭 구하기
df['BandWidth'] = (df['close'] - df['lower']) / df['MA20'] * 100
df = df[19:]

# 볼린저 밴드 그래프 그리기
plt.figure(figsize=(9, 8))
plt.subplot(2, 1, 1)
plt.plot(df.index, df['close'], color='#0000ff', label='Close')
plt.plot(df.index, df['upper'], 'r--', label = 'Upper band')
plt.plot(df.index, df['MA20'], 'k--', label='Moving average 20')
plt.plot(df.index, df['lower'], 'c--', label = 'Lower band')
plt.fill_between(df.index, df['upper'], df['lower'], color='0.9')
plt.title('NAVER Bollinger Band(20 day, 2 std)')
plt.legend(loc='best')

# usbplot - 볼린저 밴드 밴드폭 그래프 그리기
plt.subplot(2, 1, 2)
plt.plot(df.index, df['BandWidth'], color='m', label='BandWidth')
plt.grid(True)
plt.legend(loc='best')
plt.show()



