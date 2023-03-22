#NovusLive

from flask import Flask, render_template, Response
import cv2
import numpy as np

#path to classifiers
path = 'C:/Users/mrspe/OneDrive - Waterford Institute of Technology/Desktop/Local Github/opencv/data/haarcascades/'

#get image classifiers
face_cascade = cv2.CascadeClassifier(path +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path +'haarcascade_eye.xml')

#read image
cat = cv2.imread('catfilter.png')

#get shape of cat filter
original_cat_h,original_cat_w,cat_channels = cat.shape

#convert to gray
cat_gray = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)

#create mask and inverse mask of cat
ret, original_mask = cv2.threshold(cat_gray, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

#read video
cap = cv2.VideoCapture(0)
ret, img = cap.read()
img_h, img_w = img.shape[:2]


while True:
    
    #read each frame of video and convert to gray
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #find faces in image using classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #for every face found:

    for (x,y,w,h) in faces:
        #retangle for testing purposes
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        #cat size in relation to face by scaling
        cat_width = int(1.5 * face_w)
        cat_height = int(cat_width * original_cat_h *1.25 / original_cat_w)
        
        #setting location of coordinates of cat
        cat_x1 = face_x2 - int(face_w/2) - int(cat_width/2)
        cat_x2 = cat_x1 + cat_width
        cat_y1 = face_y1 - int(face_h*1.25)
        cat_y2 = cat_y1 + cat_height 

        #check to see if out of frame
        if cat_x1 < 0:
            cat_x1 = 0
        if cat_y1 < 0:
            cat_y1 = 0
        if cat_x2 > img_w:
            cat_x2 = img_w
        if cat_y2 > img_h:
            cat_y2 = img_h

        #Account for any out of frame changes
        cat_width = cat_x2 - cat_x1
        cat_height = cat_y2 - cat_y1

        #resize cat to fit on face
        cat = cv2.resize(cat, (cat_width,cat_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (cat_width,cat_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (cat_width,cat_height), interpolation = cv2.INTER_AREA)

        #take ROI for cat from background that is equal to size of cat image
        roi = img[cat_y1:cat_y2, cat_x1:cat_x2]

        #original image in background (bg) where cat is not present
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(cat,cat,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        img[cat_y1:cat_y2, cat_x1:cat_x2] = dst

        break

    cv2.imshow('img',img) #display image
    
        #if user pressed 'q' break
    if cv2.waitKey(1) == ord('q'): # 
        break

cap.release() #turn off camera 
cv2.destroyAllWindows() #close all windows





app = Flask(__name__)

# camera = cv2.VideoCapture(0)  # use 0 for web camera

# def gen_frames():  # generate frame by frame from camera
#     while True:
#         # Capture frame-by-frame
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


# @app.route('/video_feed')
# def video_feed():
#     #Video streaming route. Put this in the src attribute of an img tag
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/')
# def index():
#     #Video streaming Home Page
#     return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

#mux

