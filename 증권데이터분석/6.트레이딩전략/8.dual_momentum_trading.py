import pandas as pd
import pymysql
from datetime import datetime
from datetime import timedelta
import sys
sys.path.append(r'C:\Users\kwonk\Downloads\개인 프로젝트\juno1412-1\증권데이터분석\DB_API')
import Analyzer 

class DualMomentum:
    def __init__(self):
        """생성자: KRX 종목코드(codes)를 구하기 위한 MarkgetDB 객체 생성"""
        self.mk = Analyzer.MarketDB()
    
    def get_rltv_momentum(self, start_date, end_date, stock_count):
        """특정 기간 동안 수익률이 제일 높았던 stock_count 개의 종목들 (상대 모멘텀)
            - start_date  : 상대 모멘텀을 구할 시작일자 ('2020-01-01')   
            - end_date    : 상대 모멘텀을 구할 종료일자 ('2020-12-31')
            - stock_count : 상대 모멘텀을 구할 종목수
        """       

        # 1. daily_price 테이블에서 사용자가 입력한 일자와 같거나 작은 일자를조회해 실제 거래일을 구한다.
        connection = pymysql.connect(host='localhost', port=3307, db='INVESTAR', user='root', passwd='mariadb', autocommit=True)
        cursor = connection.cursor()

        sql = f"select max(date) from daily_price where date <= '{start_date}'"
        cursor.execute(sql)
        result = cursor.fetchone()

        if (result[0] is None):
            print("start_date : {} -> returned None".format(sql))
            return
        # 2. DB에서 조회된 거래일을 %Y-%m-%d 포맷 문자열로 변환해 사용자가 입력한 조회 시작일자 변수에 반영
        start_date = result[0].strftime('%Y-%m-%d')


        sql = f"select max(date) from daily_price where date <= '{end_date}'"
        cursor.execute(sql)
        result = cursor.fetchone()

        if (result[0] is None):
            print("end_date : {} -> returned None".format(sql))
            return
        end_date = result[0].strftime('%Y-%m-%d')

        # 3. 종목별 수익률 계산
        # 상대 모멘텀은 종목별 수익률을 구하는 것
        rows = []
        columns = ['code', 'company', 'old_price', 'new_price', 'returns']
        for _, code in enumerate(self.mk.codes):
            sql = f"select close from daily_price"\
                f"where code='{code}' and date'{start_date}"
            
            cursor.execute(sql)
            result = cursor.fetchone()
            
            if (result is None):
                continue
            old_price = int(result[0])
            sql = f"select close from daily_price"\
                f"where code='{code}' and date'{end_date}"
            cursor.execute(sql)
            result = cursor.fetchone()
            if (result is None):
                continue

            new_price = int(result[0])
            returns = (new_price / old_price - 1) * 100
            rows.append([code, self.mk.close[code], old_price, new_price, returns])


            # 상대 모멘텀 데이터 프레임 생성
            # 2차원 리스트에 저장한 종목별 수익률을 데이터프레임으로 변환해 수익률이 높은 순서로 출력한다
            df = df.DataFrame(rows, columns=columns)
            df = df[['code', 'company', 'old_price', 'new_price', 'returns']]
            df = df.sort_values(by='returns', ascending = False)
            df = df.head(stock_count)
            df.index = pd.Index(range(stock_count))

            connection.close()
            print(df)
            print(f"\nRelative Momentun ({start_date} ~ {end_date}) :"\
                f"{df['returns'].mean():.2f}% \n")
            
            return df 
        


    def get_abs_momentum(self, rltv_momentum, start_date, end_date):
        """특정 기간 동안 상대 모멘텀에 투자했을 때의 평균 수익률 (절대 모멘텀)
            - rltv_momentum : get_rltv_momentum() 함수의 리턴값 (상대 모멘텀)
            - start_date    : 절대 모멘텀을 구할 매수일 ('2020-01-01')   
            - end_date      : 절대 모멘텀을 구할 매도일 ('2020-12-31')
        """