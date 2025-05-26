# server_with_cors.py
from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        return super().end_headers()

httpd = HTTPServer(('127.0.0.1', 8888), CORSRequestHandler)
print("Serving with CORS on http://127.0.0.1:8888")
httpd.serve_forever()
