#Snap-Live Livestreaming

from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np

app = Flask(__name__)
socketioApp = SocketIO(app)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def gen_frames():

    #get image classifiers
    face_cascade = cv2.CascadeClassifier('HaarCascades/haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier(path +'haarcascade_eye.xml')

    while True:
        #read each frame of video and convert to gray
        ret, img = cap.read()
        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #find faces in image using classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        ##print(len(faces)) ##Check amount of faces on screen
        

    #for every face found:


        for (x,y,w,h) in faces:
            #retangle for testing purposes
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            #coordinates of face region
            face_w = w
            face_h = h
            face_x1 = x
            face_y1 = y


            break

        #cv2.imshow('img',img) #display image
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result
        
            #if user pressed 'q' break
        if cv2.waitKey(1) == ord('q'): # 
            break

    cap.release() #turn off camera  
    cv2.destroyAllWindows() #close all windows


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    #Video streaming Home Page
    
    return render_template('index.html')

def run():
    socketioApp.run(app)

if __name__ == '__main__':
    socketioApp.run(app)


