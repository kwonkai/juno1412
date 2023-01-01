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

# 30일 기준 현금흐름 구하기
# 1. 종가 구하기 : 고가, 저가, 종가의 합을 3으로 나누기
df['PB'] = (df['close'] + df['lower']) / (df['upper'] + df['lower'])
df['TP'] = (df['high'] + df['low'] + df['close']) / 3
df['PMF'] = 0 # 긍정적 현금흐름
df['NMF'] = 0 # 부정적 현금흐름

# 2. range함수로 마지막 값을 포함하지 않으므로 0 ~ -2까지 반복한다.
for i in range(len(df.close) -1):
    if df.TP.values[i] < df.TP.values[i+1]:

        # 3. i번째 중심가격보다 i+1 번째 중심가격이 높으면, i+1번재 중심가격과 i+1번째 거래량의 곱을 i+1번째 긍정적 현금흐름에 저장한다
        df.PMF.values[i+1] = df.TP.values[i+1] * df.volume.values[i+1]
        df.NMF.values[i+1] = 0

    else:
        df.NMF.values[i+1] = df.TP.values[i+1] * df.volume.values[i+1]
        df.PMF.values[i+1] = 0

# 4. 30일 동안의 긍정적 현금흐름 합을 부정적 현금흐름의 합을 나눈 결과를 MFR(현금흐름 비율)에 저장한다.
df['MFR'] = df.PMF.rolling(window=30).sum() / df.NMF.rolling(window=30).sum()

# 5. 30일 기준으로 현금흐름 지수를 계산한 결과를 MFI30에 저장한다.
df['MFI30'] = 100 - 100 / (1 + df['MFR'])



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

for i in range(len(df.close)):
    # 6. %b가 0.8보다 크고 30일 기준 MFI가 80보다크면, 매수시점을 빨간색 삼각형으로 표시한다.
    if df.PB.values[i] > 0.8 and df.MFI30.values[i] > 80:
        plt.plot(df.index.values[i], df.close.values[i], 'r^')
    
    # 7. %b가 0.2보다 작고 30일 기준 MFI가 20보다작으면, 매도시점을 파란색 삼각형으로 표시한다.
    elif df.PB.values[i] < 0.2 and df.MFI30.values[i] < 20:
        plt.plot(df.index.values[i], df.close.values[i], 'bv')  

plt.title('NAVER Bollinger Band(30 day, 2 std)')
plt.legend(loc='best')

# subplot - 볼린저 밴드 밴드폭 그래프 그리기
plt.subplot(2, 1, 2)
# 8.MFI와 비교할 수 있게 %b를 그대로 표시하지 않고 100을 곱해 푸른색 실선으로 표시한다.
plt.plot(df.index, df['PB'] * 100, 'b', label='%B X 100')

# 9. 30일 기준 MFI를 녹색의 점선으로 표기한다.
plt.plot(df.index, df['MFI30'] * 100, 'g--', label='%B X 100')

# 10. Y축 눈금을 -20부터 120까지 20단위로 표시한다.
plt.yticks([-20, 0, 20, 40, 60, 80, 100, 120])
for i in range(len(df.close)):
    if df.PB.values[i] > 0.8 and df.MFI30.values[i] > 80:
        plt.plot(df.index.values[i], df.close.values[i], 'r^')
    elif df.PB.values[i] < 0.2 and df.MFI30.values[i] < 20:
        plt.plot(df.index.values[i], df.close.values[i], 'bv')  
plt.grid(True)
plt.legend(loc='best')
plt.show()



