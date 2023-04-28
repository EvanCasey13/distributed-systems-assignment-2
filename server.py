import eventlet
from livevideo import app
import socketio
from waitress import serve
from livevideo import run

sio = socketio.Server()
appServer = socketio.WSGIApp(sio, app)

@sio.event
def connect(sid, environ):
    print('connect ', sid)

@sio.event
def my_message(sid, data):
    print('message ', data)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    serve(app=app, host='0.0.0.0', port=5000, url_scheme='http', threads=6)

