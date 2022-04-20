from http.server import BaseHTTPRequestHandler, HTTPServer # python3
import socketserver
import json
import cgi

class HandleRequests(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    
    # get으로 메세지 보내기
    def do_GET(self):
        self._set_headers()
        self.wfile.write("received get request")
    
    # post로 json 폴더에 메세지 보내기
    def do_POST(self):
        
        '''Reads post request body'''
        self._set_headers()
        content_len = int(self.headers.getheader('content-length', 0))
        post_body = self.rfile.read(content_len)
        self.wfile.write("received post request:<br>{}".format(post_body))

    def do_PUT(self):
        self.do_POST()

host = ''
port = 8082
HTTPServer((host, port), HandleRequests).serve_forever()

