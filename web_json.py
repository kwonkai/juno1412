
import json
# json 만들기
fileName = "my-data.json"
jsonString = '{"type": "title", "id": "581, 573, 606, 115, 591, 223, 549, 402, 203, 550" \
               "type": "learningArea", "id": "573, 115, 606, 591, 581, 223, 86, 618, 592, 589"\
               "type": "level", "id": "115, 606, 591, 581, 573, 223, 558, 533, 394, 592"}'


with open(fileName, "w") as f:
    jsonString = json.dump(jsonString, f, indent='\n', sort_keys=True)

# jsonString = json.loads(jsonString)


# SorketServer 모듈 = 네트워크 설정 시 필요한 클래스와 기능 제공
# SorketServer모듈의 TCPServer는 TCP프로토콜을 사용해 서버를 설정한다. 생성자는 서버주소, 서버요청 클래스(튜플) 허용
# Simplt
from email import message
from http.server import BaseHTTPRequestHandler, HTTPServer # python3
import socketserver
import json
import cgi
from xmlrpc.client import Server

# sorketserver : 네트워크 서버를 위한 프레임워크pyth
# cgi : 공용 게이트웨이 인터페이스

class HandleRequests(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write("Hello".encode())

    def do_HEAD(self):
        self._set_headers()

    
    # get으로 메세지 보내기
    def do_GET(self):
        self._set_headers()
        self.wfile.write(json.dumps({jsonString}))

    
    # post로 json 폴더에 메세지 보내기
    def do_POST(self):
        ctype = cgi.parse_header(self.headers.getheader('content-type'))
        
        # json 파일이 아니라면 거부
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            return

        # 메세지 리딩 -> python dictionary으로 전환
        jsonString = self.headers.getheader('content-length')
        message = json.loads(self.rfile.read(jsonString))

        # 데이터 객체에 속성 추가
        message['received'] = 'ok'

        # 메세지 회신
        self._set_headers()
        self.wfile.write(json.dumps(message))

# http server run
def run(server_class = HTTPServer, handler_class=Server, port = 8081):
    server_address = ('', port)
    # httpd = http deamon, 웹서버 배그라운드에서 실행되어 들어오는 서버 요청을 대기하는 역할
    httpd = server_class(server_address, handler_class)

    print('server starting') 
    httpd.serve_forever()


# # 서버 run
# if __name__ == "__main__":
#     from sys import argv

#     if len(argv) == 2:
#         run(port=int(argv[1]))
#     else:
#         run()


