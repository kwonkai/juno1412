import pandas as pd
from bs4 import BeautifulSoup
import pymysql, calendar, time, json
import requests
from datetime import datetime
from threading import Timer

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
            url = f"http://finance.naver.com/item/sise_day.nhn?code={code}"
            html = BeautifulSoup(requests.get(url,
                headers={'User-agent': 'Mozilla/5.0'}).text, "lxml")
            pgrr = html.find("td", class_="pgRR")
            if pgrr is None:
                return None
            s = str(pgrr.a["href"]).split('=')
            lastpage = s[-1] 
            df = pd.DataFrame()
            pages = min(int(lastpage), pages_to_fetch)
            for page in range(1, pages + 1):
                pg_url = '{}&page={}'.format(url, page)
                df = df.append(pd.read_html(requests.get(pg_url,
                    headers={'User-agent': 'Mozilla/5.0'}).text)[0])                                          
                tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                print('[{}] {} ({}) : {:04d}/{:04d} pages are downloading...'.
                    format(tmnow, company, code, page, pages), end="\r")
            df = df.rename(columns={'날짜':'date','종가':'close','전일비':'diff'
                ,'시가':'open','고가':'high','저가':'low','거래량':'volume'})
            df['date'] = df['date'].replace('.', '-')
            df = df.dropna()
            df[['close', 'diff', 'open', 'high', 'low', 'volume']] = df[['close',
                'diff', 'open', 'high', 'low', 'volume']].astype(int)
            df = df[['date', 'open', 'high', 'low', 'close', 'diff', 'volume']]
        
        except Exception as e:
            print('Exception occured :', str(e))
            return None
        
        return df






    # 네이버 금융에서 읽어온 주식 시세를 DB에 REPLACE
    def replace_into_db(self, df, num, code, company):
        with self.conn.cursor() as curs:
            # 1. 인수로 넘겨받은 dataframe을 tuple로 순회처리한다.
            for r in df.itertuples():
                # 2. REPLACE INTO 구문으로 daily_price 테이블 업데이트
                # 값이 string 이면 '{}', int 이면 {}
                sql = f"REPLACE INTO daily_price VALUES ('{code}', "\
                    f"'{r.date}', {r.open}, {r.high}, {r.low}, {r.close}, "\
                    f"{r.diff}, {r.volume})"
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
        # 1. update_comp_info() method를 호출 해 상장법인 목록을 DB에 업데이트 한다.
        self.update_comp_info()

        # 2. DBUpdater.py가 있는 디렉터리에서 config.json 파일을 읽기모드로 열어준다.
        # 3. 파일이 있다면 page_to_fetch 값을 읽어서 사용한다.
        try:
            with open('config.json','r') as in_file:
                config = json.load(in_file)
                pages_to_fetch = config['pages_to_fetch']
        
        # 4. 파일이 없다면 config.json 파일을 생성해준다. 처음 생성 시 page_to_fetch 100, 이후 1로 설정
        except FileNotFoundError:
            with open('config.json', 'w') as out_file:
                pages_to_fetch = 100
                config = {'pages_to_fetch': 1}
                json.dump(config, out_file)

        # 5. pages_to_fetch 값으로 update_daily_price method를 호출한다.
        self.update_daily_price(pages_to_fetch)

        # 6. 이번달 마지막날(lastday)을 구해 다음날 오후 5시를 계산한다.
        tmnow = datetime.now()
        lastday = calendar.monthrange(tmnow.year, tmnow.month)[1]
        if tmnow.month == 12 and tmnow.day == lastday:
            tmnext = tmnow.replace(year=tmnow.year+1, month=1, day=1, hour=17, minute=0, second=0)
        elif tmnow.month == lastday:
            tmnext = tmnow.replace(month=tmnow.month+1, day=1, hour=17, minute=0, second=0)
        else:
            tmnext = tmnow.replace(day = tmnow.day+1, hour=17, minute=0, second=0)
        
        tmdiff = tmnext - tmnow
        secs = tmdiff.seconds

        # 7. 다음날 오후 5시에 excute_daily() method를 실행하는 타이머 객체를 설정한다.
        t = Timer(secs, self.execute_daily)
        print("Waiting for next update ({}) ...".format(tmnext.strftime('%Y-%m-%d %H:%M')))

        t.start()
        





if __name__ == '__main__':
    dbu = DBUpdater()
    dbu.execute_daily()
