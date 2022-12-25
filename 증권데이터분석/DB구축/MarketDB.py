# 네이버 일별 시세 API

# 라이브러리 설정
import pymysql
import pandas as pd
from datetime import datetime
from datetime import timedelta
import re

class MarketDB:
    # 생성자 : MariaDB 연결 및 종목코드 딕셔너리 생성
    def __init__(self):
        self.conn = pymysql.connect(host='localhost', port='3307', user='root', \
                                    password='mariadb', db ='INVESTAR', charset='utf8')
        self.codes = {}
        self.get_comp_info()

    # 소멸자 : MariaDB 연결 해제
    def __del__(self):
        self.conn.close()

    # company_info 테이블에서 읽어와 codes에 저장
    def get_comp_info(self):
        sql = "SELECT * FROM company_info"

        companyInfo = pd.read_sql(sql, self.conn)
        for idx in range(len(companyInfo)):
            self.codes[companyInfo['code'].values[idx]] = companyInfo['company'].values[idx]

    # KRX 종목의 일별 시세를 데이터프레임 형태 변환
    def get_daily_price(self, code, start_date=None, end_date=None):
        if start_date is None:
            one_year_ago = datetime.today() - timedelta(daus=365)

        # 1. pandas read_sql()함수를 이용해 SQL 구문의 결과를 데이터 프레임으로 가져온다.
        # 데이터 프레임으로 가져오면 정수형 인덱스가 별도로 생성됨
        sql = "SELECT * FROM daily_price WHERE code = '{}' and date >= '{}' and date <= '{}'".format(code, start_date,end_date)
        df = pd.read_sql(sql, self.conn)
        
        # 2. 데이터프레임의 인덱스를 df의 'date'칼럼으로 새로 설정된다.
        df.index = df['date']

        return df