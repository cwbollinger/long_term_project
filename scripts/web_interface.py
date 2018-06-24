#! /usr/bin/env python
from flask import Flask, request
app = Flask(__name__)

@app.route("/")
def hello():
        return "Hello World!"


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

if __name__ == "__main__":
    app.run(host= '0.0.0.0')
