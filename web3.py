from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib
import json

# HTTP Request handlers
class ncbot_RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        dict = json.loads(post_data)
        print(dict.get('a'))
        return

def run():
    print("Starting a server")
    server_address = ("127.0.0.1", 8888)
    httpd = HTTPServer(server_address, ncbot_RequestHandler)

    print("Running the server")
    httpd.serve_forever()

run()