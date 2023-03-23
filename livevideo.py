#Facial recognition application
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import face_recognition

app = Flask(__name__)
socketioApp = SocketIO(app)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load a third sample picture and learn how to recognize it.
evan_image = face_recognition.load_image_file("evan.jpg")
evan_face_encoding = face_recognition.face_encodings(evan_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    evan_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Evan Casey"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():

    #get image classifiers
    face_cascade = cv2.CascadeClassifier('HaarCascades/haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier(path +'haarcascade_eye.xml')

    while True:
        #read each frame of video and convert to gray
        ret, img = cap.read()
         # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #find faces in image using classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        ##print(len(faces)) ##Check amount of faces on screen
        #for every face found:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            
        for (x,y,w,h) in faces:
            #retangle for testing purposes
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            # Draw a label with a name above the face
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)

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


