#! /usr/bin/env python
import os
import BaseHTTPServer
import SimpleHTTPServer
server_address = ("", 8080)

class MyRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if self.path == '/log_data/':
            return os.environ['HOME'] + '/bags'
        else:
            return SimpleHTTPServer.SimpleHTTPRequestHandler.translate_path(self, path)

# figure out where the ros2djs folder is relative to this file location
fileloc = os.path.abspath(__file__)
folder = os.path.dirname(fileloc)
new_home_dir = os.path.abspath(os.path.join(folder, '..', 'ros2djs'))
print(new_home_dir)
os.chdir(new_home_dir)
httpd = BaseHTTPServer.HTTPServer(server_address, MyRequestHandler)
httpd.serve_forever()
