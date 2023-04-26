#Facial recognition application
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import face_recognition
from threading import Thread
import math

app = Flask(__name__)
socketioApp = SocketIO(app)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load a sample picture and learn how to recognize it.
dustin_image = face_recognition.load_image_file("dustin_p.png")
dustin_face_encoding = face_recognition.face_encodings(dustin_image)[0]

# Load a second sample picture and learn how to recognize it.
alex_image = face_recognition.load_image_file("alex_pereira.png")
alex_face_encoding = face_recognition.face_encodings(alex_image)[0]

# Load a third sample picture and learn how to recognize it.
evan_image = face_recognition.load_image_file("evan.jpg")
evan_face_encoding = face_recognition.face_encodings(evan_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    dustin_face_encoding,
    alex_face_encoding,
    evan_face_encoding
]
known_face_names = [
    "Dustin Porier",
    "Alex Pereira",
    "Evan Casey"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

def face_acc_percent(face_distance, face_match_threshold=0.6):
       range = (1.0 - face_match_threshold)
       linear_val = (1.0 - face_distance) / (range * 2.0)

       if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + '%'
       else:
          value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
       return str(round(value, 2)) + '%'

def face_model(img):
        process_this_frame = True
 # Only process every other frame of video to save time
        if process_this_frame:

         # Resize frame of video to 1/4 size for faster face recognition processing
         small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

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
                confidence = "?"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    confidence = face_acc_percent(face_distances[best_match_index], face_match_threshold=0.6)

                face_names.append(name + " " + confidence)

                process_this_frame = not process_this_frame
            
        for (x,y,w,h), name in zip(face_locations, face_names):
            x *= 4
            y *= 4
            w *= 4
            h *= 4
            #retangle for testing purposes
            cv2.rectangle(img,(h,x),(y,w),(255,0,0),2)

            # Draw a label with a name above the face
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (h + 5 , x - 35), font, 1.0, (255, 255, 255), 1)

            break

def gen_frames():
    while True:
        #read each frame of video and convert to gray
        ret, img = cap.read()

        fModel_thread = Thread(target=face_model(img), args=())
        fModel_thread.start()

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

genF_thread = Thread(target=gen_frames, args=())
genF_thread.start()