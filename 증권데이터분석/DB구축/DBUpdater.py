# 라이브러리 설정
import enum
from bs4 import BeautifulSoup
import pymysql
import pandas as pd
from datetime import datetime

import requests

class DBUpdater:
    # 생성자 = MariaDB 연결 및 종목코드 딕셔너리 생성
    def __init__(self):
        self.conn = pymysql.connect(host='localhost', port=3307, user='root',
                                    password='mariadb', db='INVESTAR', charset='utf8')
        
        with self.conn.cursor() as curs:
            sql = """
            CREATE TABLE IF NOT EXISTS company_info (
                code VARCHAR(20),
                company VARCHAR(40),
                last_update DATE,
                PRIMARY KEY(CODE))
            """
            curs.execute(sql)

            sql = """
            CREATE TABLE IF NOT EXISTS daily_price (
                code VARCHAR(20),
                date DATE,
                open BIGINT(20),
                high BIGINT(20),
                low BIGINT(20),
                close BIGINT(20),
                diff BIGINT(20),
                volume BIGINT(20),
                PRIMARY KEY (code, date))
            """
            curs.execute(sql)
        self.conn.commit()

        self.codes = dict()
        self.update_comp_info()
    
    # 소멸자 : MariaDB 연결 해제
    def __del__(self):
        self.conn.close()

    # KRX로부터  상장법인 목록 파일을 읽어와 데이터 프레임 변환
    def read_krx_code(self):
        # krx 상장법인목록 url 가져와서 read_html로 읽기
        url = 'https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
        krx = pd.read_html(url, header = 0)[0]

        # krx 상장법인목록 columns 중 종목코드, 회사명만 가져오기
        krx = krx[['종목코드', '회사명']]
        
        # krx 칼럼을 종목코드 -> code로, 회사명을 company로 변경
        krx = krx.rename(columns={'종목코드':'code', '회사명':'company'})

        # krx 종목코드 6자리에 빠진 0을 추가해준다.
        krx.code = krx.code.map('{:06d}'.format)

        return krx # krx를 반환

    # 종목코드를 conpany_info 테이블에 업데이트 한 후 딕셔너리에 저장
    # 오늘 날짜로 업데이트한 기록이 있다면 더이상 업데이트 하지 않음
    def update_comp_info(self):
        """종목코드를 company_info 테이블에 업데이트 한 후 딕셔너리에 저장"""
        sql = "SELECT * FROM company_info"
        df = pd.read_sql(sql, self.conn)
        for idx in range(len(df)):
            self.codes[df['code'].values[idx]] = df['company'].values[idx]
                    
        with self.conn.cursor() as curs:
            sql = "SELECT max(last_update) FROM company_info"
            curs.execute(sql)
            rs = curs.fetchone()
            today = datetime.today().strftime('%Y-%m-%d')
            
            if rs[0] == None or rs[0].strftime('%Y-%m-%d') < today:
                krx = self.read_krx_code()
                for idx in range(len(krx)):
                    code = krx.code.values[idx]
                    company = krx.company.values[idx]
                    with self.conn.cursor() as curs:                
                        sql = f"REPLACE INTO company_info (code, company, last"\
                        f"_update) VALUES ('{code}', '{company}', '{today}')"
                        curs.execute(sql)
                        self.codes[code] = company
                        tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                        print(f"[{tmnow}] #{idx+1:04d} REPLACE INTO company_info "\
                            f"VALUES ({code}, {company}, {today})")
                    self.conn.commit()
                    print('')              


    # 네이버금융에서 주식 시세를 읽어서 데이터프레임으로 변환
    def read_naver(self, code, company, pages_to_fetch):
        try:
            url = f'https://finance.naver.com/item/sise_day.nhn?code={code}'
            html = requests.get(url, headers={'User-agent': 'Mozila/5.0'})
            bs = BeautifulSoup(html, "lxml")
            pgrr = bs.find("td", class_="pgRR")
            if pgrr is None:
                return None
            
            s = str(pgrr.a["href"]).split('=')

            # 1. 네이버 금융 일별 시세의 마지막 페이지 탐색
            lastpage = s[-1]
            df = pd.DataFrame()

            # 2. 설정파일에 설정된 페이지수(pages_to_fetch)와 1의 페이지 수에서 작은 것을 선택
            pages = min(int(lastpage), pages_to_fetch)

            # 3. 일별 시세 페이지를 read_html()로 읽어 데이터 프레임에 추가
            for page in range(1, pages+1):
                url = '{}&page={}'.format(url, page)
                req = requests.get(url, headers={'User-agent': 'Mozila/5.0'})
                df = df.append(pd.read_html(req.text, header=0)[0])
                tmnow = datetime.now().strftime('&Y-%m-%d %H %M')
                print('[{}] {} ({}) : {:04d}/{:04d} pages arae downloading...'.format(tmnow, company, code, page, pages), end="\r")
                
                # 4. df dataframe의 columns 명 변경
                df = df.rename(columns = {'날짜':'date', '종가':'close', '전일비':'diff', '시가':'open', '고가':'high', '저가':'low', '거래량':'volume'})                
                # 5. 연,월,일 형식의 date 데이터를 연-월-일 형식으로 변경
                df['date'] = df['date'].replace('.', '-')
                # 6. 결측치 제거
                df = df.dropna()
                # 7. df 데이터프레임 value 값 str -> int로 변경
                df[['close', 'diff', 'open', 'high', 'low','volume']] = df[['close', 'diff', 'open', 'high', 'low','volume']].astype(int)

                # 8.시간, OHLC, DIFF, 거래량만 가져오기
                df = df[['open', 'high', 'low', 'close', 'diff', 'volume']]
        
        except Exception as e:
            print('Exception occured :', str(e))
            return None
        
        return df






    # 네이버 금융에서 읽어온 주식 시세를 DB에 REPLACE
    def replace_into_db(self, df, num, code, company):
        with self.conn.consor() as curs:
            # 1. 인수로 넘겨받은 dataframe을 tuple로 순회처리한다.
            for r in df.itertuples():
                # 2. REPLACE INTO 구문으로 daily_price 테이블 업데이트
                # 값이 string 이면 '{}', int 이면 {}
                sql = f"REPLACE INTO daily_price VALUES"\
                    f"('{code}', '{r.date}', {r.open}, {r.high}, {r.low}, {r.close}" \
                    f"{r.diff}, {r.volume}"
                curs.execute(sql)
            
            # 3. commit() 함수로 maria DB에 반영한다.
            self.conn.commit()
            print('[{}] #{:04d} {} ({}) : {} rows > REPLCE INTO daily_price [OK]'\
                    .format(datetime.now().strftime('%Y-%m-%d %H:%M'), num+1, company, code, len(df)))

    
    # KRX 상장 법인의 주식 시세를 네이버로 부터 읽어 DB에 업데이트
    def update_daily_price(self, pages_to_fetch):
        # 1. self.codes 딕셔너리에 저장된 종목코드에 대한 순회처리 및 numbering
        for idx, code in enumerate(self.codes):
            # 2. read_naver() method를 이용해 종목코드에 대한 일별 시세 데이터의 dataframe 구하기
            df = self.read_naver(code, self.codes[code], pages_to_fetch)
            
            # df가 None이라도 계속 진행
            if df is None:
                continue

            # 3. 일별 시세 데이터프레임이 구해지면 replace_into_db method로 DB저장
            self.replace_into_db(df, idx, code, self.codes[code])


    # 실행 즉시 매일 오후 5시에 daily_price 테이블 업데이트
    def execute_daily(self):

if __name__ == '__main__':
    dbu = DBUpdater()
    dbu.update_comp_info()
