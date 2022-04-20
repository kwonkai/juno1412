from http.server import HTTPServer, SimpleHTTPRequestHandler

httpd = HTTPServer( ('127.0.0.1', 8080), SimpleHTTPRequestHandler )
print( '서버시작' )
httpd.serve_forever()
print( '서버종료' )

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



