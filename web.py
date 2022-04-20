from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

class myHTTPRequestHandler(BaseHTTPRequestHandler):
    # get형식 파라미터 파싱하기 = url "?" 이후의 값을 딕셔너리로 파싱
    def do_GET(self, key):
        print('get방식 요청')
        if hasattr(self, "_myHandler__param") == False:
            if "?" in self.path:
                # url의 "?"이후 값 파싱
                self.__param = dict(urlparse.parse_qsl(self.path.split("?")[1], True));
            else :
                # url의 "?"가 없으면 빈 딕셔너리를 넣음
                self.__param = {};
            
        if key in self.__param:
            return self.__param[key];
        return None;
    
    # http 프로토콜의 header 내용
    def __set_Header(self, code):
        # 응답 코드를 파라미터로 받아 응답
        self.send_response(code);
        self.send_header('Content-type', 'text/html')
        self.end_headers();

    # http 프로토콜 body 내용
        


        # self.send_response(200)
        # self.send_header( 'Contend-type', 'text/html; charset=utf-8')
        # self.end_headers()

        # parts = urlparse( self.path)
        # keyword, value = parts.query.split( '=', 1)

        # self.send_body

    # 전송데이터 method화
    def send_body1(self):
        html='''
        <!doctype html>
        <html lang="ko">
        <head>
        </head>
        <body>
        Hello Python
        </body>
        </html>   
        '''
        self.wfile.write( html.encode('utf-8'))

        #전송 데이터 메서드화
    def send_body2(self):
        html='''
        <!doctype html>
        <html lang="ko">
        <head>
        <meta charset="utf-8" />
        </head>
        <body>
        Hello Python, 안녕 난 파이썬2
        </body>
        </html>
        '''
        self.wfile.write( html.encode( 'utf-8' ) )




if __name__ == '__main__':
    httpd = HTTPServer( ('127.0.0.1', 8080), SimpleHTTPRequestHandler )
    print( '서버시작' )
    httpd.serve_forever()
    print( '서버종료' )


# class myHTTPRequestHandler(BaseHTTPRequestHandler):
#     # get형식 파라미터 파싱하기 = url "?" 이ㅜㅎ의 값을 딕셔너리로 파싱
#     def do_GET(self):
#         print('get방식 요청')
#         self.send_response(200)
#         self.send_header( 'Contend-type', 'text/html; charset=utf-8')
#         self.end_headers()

#         parts = urlparse( self.path)
#         keyword, value = parts.query.split( '=', 1)

#         self.send_body

#     # 전송데이터 method화
#     def send_body1(self):
#         html='''
#         <!doctype html>
#         <html lang="ko">
#         <head>
#         </head>
#         <body>
#         Hello Python
#         </body>
#         </html>   
#         '''
#         self.wfile.write( html.encode('utf-8'))

#         #전송 데이터 메서드화
#     def send_body2(self):
#         html='''
#         <!doctype html>
#         <html lang="ko">
#         <head>
#         <meta charset="utf-8" />
#         </head>
#         <body>
#         Hello Python, 안녕 난 파이썬2
#         </body>
#         </html>
#         '''
#         self.wfile.write( html.encode( 'utf-8' ) )







# import http.server
# import socketserver

# class myHandler(BaseHTTPRequestHandler):
#     def do_GET(self):
#         self.send_response(200)
#         self.send_header('Content-type','text/html')
#         self.end_headers()
#         self.wfile.write('hello\n'.encode())
#         return


# print('Server listening on port 8000...')
# httpd = HTTPServer(('',8000), myHandler)
# httpd.serve_forever()



