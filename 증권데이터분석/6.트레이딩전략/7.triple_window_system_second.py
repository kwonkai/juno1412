# 라이브러리 설정

# 모듈경로 지정
import sys
import matplotlib
sys.path.append(r'C:\Users\kwonk\Downloads\개인 프로젝트\juno1412-1\증권데이터분석\DB_API')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axis import Axis 

import datetime
from mplfinance.original_flavor import candlestick_ohlc

import Analyzer

# 1. DB가져오기
mk = Analyzer.MarketDB()
df = mk.get_daily_price('케이티', '2018-01-01')

# 2. 첫번째창 - 지수이동평균, MACD선, 신호선, MACD히스토그램 구하기
# 종가 12주 지수, 24주 이동 평균5
ema60 = df.close.ewm(span=60).mean()
ema130 = df.close.ewm(span=130).mean()
# MACD선 구하기
macd = ema60 - ema130
# 신호선(MACD의 9주 지수 이동평균 구하기)
signal = macd.ewm(span=45).mean()
# MACD 히스토그램 = MACD선 - 신호선
macdhist = macd - signal


# 3. 캔들차트에 사용될 수 있도록 날짜(date)형 인덱스를 숫자형으로 변경
df = df.assign(ema130 = ema130, ema60 = ema60, macd=macd, signal=signal, macdhist=macdhist).dropna()
df['number'] = df.index.map(mdates.date2num)
 

# 4. OHLC의 숫자형 일자, 시가, 고가, 저가, 종가 값을 이용해 캔들차트를 그린다.
ohlc = df[['number', 'open', 'high', 'low', 'close']]


## 두번째 창 - 시장 파도
ndays_high = df.high.rolling(window=14, min_periods=1).max()
ndays_low = df.high.rolling(window=14, min_periods=1).min()

fast_k = (df.close - ndays_low) / (ndays_high - ndays_low) * 100
slow_d = fast_k.rolling(window=3).mean()
df = df.assign(fast_k=fast_k, slow_d=slow_d).dropna()



# 5. 삼중창 매매시스템 - 두번째창그래프 그리기
plt.figure(figsize=(9,7))

p1 = plt.subplot(2, 1, 1)
plt.title('Title Screen Trading - Second Screen (KT)')
plt.grid(True)
candlestick_ohlc(p1, ohlc.values, width=0.7, colorup = 'red', colordown = 'blue')
p1.xaxis.set_major_formatter(formatter=mdates.DateFormatter('%Y-%m'))
plt.plot(df.number, df['ema130'], color = 'c', label='EMA130')
plt.legend(loc='best')

# 두번째 창
p2 = plt.subplot(2, 1, 2)
plt.grid(True)
p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.bar(df.number, df['fast_k'], color='c', label = '&K')
plt.plot(df.number, df['slow_d'], color='k', label = '%D')
plt.yticks([0, 20, 80, 100])
plt.legend(loc='best')
plt.show()



