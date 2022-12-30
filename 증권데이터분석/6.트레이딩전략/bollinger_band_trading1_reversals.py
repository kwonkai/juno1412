# 볼린저 밴드 - 네이버 증권 종가

# 모듈경로 지정
import sys
sys.path.append(r'C:\Users\kwonk\Downloads\개인 프로젝트\juno1412-1\증권데이터분석\DB_API')

# 라이브러리 설정
import matplotlib.pyplot as plt
import Analyzer

# NAVER 주가 종가 가져오기
mk = Analyzer.MarketDB()
df = mk.get_daily_price('삼성전자', '2022-07-01')

# 볼린저 밴드 구하기
# 1. 종가평균, 표준편차, 상단밴드, 하단밴드,%b 을 구한다.
df['MA20'] = df['close'].rolling(window=20).mean() 
df['stddev'] = df['close'].rolling(window=20).std() 
df['upper'] = df['MA20'] + (df['stddev'] * 2)
df['lower'] = df['MA20'] - (df['stddev'] * 2)
df['PB'] = (df['close'] + df['lower']) / (df['upper'] + df['lower'])

# 2. 삼성전자의 종가, 고가, 저가, 거래량으로 일중강도 II를 구한다.
df['II'] = (2*df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']

# 3. 21일간 일중강도 II 합을 21일간 거래량 합으로 나누어 일중강도 II%를 구한다.
df['IIP21'] = df['II'].rolling(window=21).sum()/df['volume'].rolling(window=21).sum()*100

# Nan 값 제거
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

for i in range(0, len(df.close)):
    # 4. %b가 0.05보다 작고, 21일 기준 II%가 0보다 크면 매수시점을 빨간색 삼각형으로 표시한다.
    if df.PB.values[i] < 0.05 and df.IIP21.values[i] > 0:
        plt.plot(df.index.value[i], df.close.values[i], 'r')
    # 5. %b가 0.96보다 크고, 21일 기준 II%가 0보다 작으면 매수시점을 파란색 삼각형으로 표시한다.
    elif df.PB.values[i] > 0.95 and df.IIP21.values[i] < 0:
        plt.plot(df.index.values[i], df.close.values[i], 'bv')


    


plt.legend(loc='best')
# subplot - 볼린저 밴드 밴드폭 그래프 그리기
plt.subplot(3, 1, 2)
plt.plot(df.index, df['PB'], 'b', label='%B')
plt.grid(True)
plt.legend(loc='best')

# 6. 3행 1열에 3번째 grid에 일중 강도율을 구한다.
plt.subplot(3, 1, 3)
plt.bar(df.index, df['IIP21'], color='g', label='II% 21day')
for i in range(0, len(df.close)):
    # 7. 세 번째 일중 강도율 그래프에서 매수 시점을 빨간색 삼각형으로 표시한다.
    if df.PB.values[i] < 0.05 and df.IIP21.values[i] > 0:
        plt.plot(df.index.values[i], 0, 'r^')
    # 8. 세 번째 일중 강도율 그래프에서 매도 시점을 파란색 삼각형으로 표시한다.
    elif df.PB.values[i] > 0.95 and df.IIP21.values[i] < 0:
        plt.plot(df.index.values[i], 0, 'bv')
plt.grid(True)
plt.legend(loc='best')
plt.show()

