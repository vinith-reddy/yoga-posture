# IMPORTS
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
import mediapipe as mp
import pandas as pd
import numpy as np
import base64
import pickle
import cv2
import os
from skimage.exposure import equalize_adapthist
import sqlite3


upload_dir = os.path.join(os.path.dirname(__file__), 'upload')
os.makedirs(upload_dir, exist_ok=True)

# INITIALIZATION
app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,
                    
                    min_detection_confidence=0.5)

# LOADING MODEL
with open('detect_pose.pkl', 'rb') as f:
    model = pickle.load(f)

# YOGA POSE DETECTION USING IMAGE
def usingImage(img_path):
    print("Processing image:", img_path)
    try:
        input_frame = cv2.imread(img_path)
        if input_frame is None:
            print("Error: Unable to read the image.")
            return ''

        result = pose.process(image=input_frame)
        if result.pose_landmarks is None:
            print("Error: No pose landmarks detected.")
            return ''

        output_image = input_frame.copy()
        label = "Unknown Pose"
        accuracy = 0

        mp_drawing.draw_landmarks(image=output_image, landmark_list=result.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        pose_landmarks = result.pose_landmarks.landmark
        row = list(np.array([[landmark.x, landmark.y, landmark.z]
                             for landmark in pose_landmarks]).flatten())
        X = pd.DataFrame([row])
        label = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]
        accuracy = str(
            round(body_language_prob[np.argmax(body_language_prob)]*100, 3))
        if(float(accuracy) < 50):
            label = "Unknown Pose"
        cv2.rectangle(output_image, (0, 0), (250, 60),
                      (0, 0, 255), 1, cv2.LINE_4)
        cv2.putText(output_image, 'Class', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
        cv2.putText(output_image, label, (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)
        cv2.putText(output_image, 'Accuracy', (150, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
        cv2.putText(output_image, accuracy, (150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)
        retval, buffer = cv2.imencode('.jpg', output_image)
        jpg_as_text = base64.b64encode(buffer)
        print("Prediction successful")
        return jpg_as_text
    except Exception as e:
        print("Error during prediction:", str(e))
        return ''

# YOGA POSE DETECTION USING WEBCAM


# Initialize the camera as a global variable
camera = cv2.VideoCapture(0)

def usingWebcam():
    global camera  # Use the global camera variable

    while True:
        if not camera.isOpened():
            # Reinitialize the camera if it's not opened
            camera = cv2.VideoCapture(0)

        status, input_frame = camera.read()
        if not status:
            break
        else:
            result = pose.process(image=input_frame)
            label = "Unknown Pose"
            accuracy = 0
            try:
                mp_drawing.draw_landmarks(image=input_frame, landmark_list=result.pose_landmarks,
                                          connections=mp_pose.POSE_CONNECTIONS)
                pose_landmarks = result.pose_landmarks.landmark
                row = list(np.array([[landmark.x, landmark.y, landmark.z]
                                     for landmark in pose_landmarks]).flatten())
                X = pd.DataFrame([row])
                label = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                accuracy = str(round(body_language_prob[np.argmax(body_language_prob)] * 100, 3))
                if float(accuracy) < 50:
                    label = "Unknown Pose"
            except:
                pass
            cv2.rectangle(input_frame, (0, 0), (250, 60), (0, 0, 255), 1, cv2.LINE_4)
            cv2.putText(input_frame, 'Class', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
            cv2.putText(input_frame, label, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)
            cv2.putText(input_frame, 'Accuracy', (150, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
            cv2.putText(input_frame, accuracy, (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_4)
        ret, buffer = cv2.imencode('.jpg', input_frame)
        input_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + input_frame + b'\r\n')

# ...



@app.route("/")
def home():
    return render_template("home.html")

@app.route("/sign")
def sign():
    return render_template("signup-in.html")

@app.route("/signup")
def signup():
    
    
    name = request.args.get('user','')
    password = request.args.get('pass','')
    password1 = request.args.get('pass1','')
    email = request.args.get('email','')
    number = request.args.get('num','')

    if password1 == password:
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("insert into `datas` (`name`, `password`,`password1`,`email`,`mobile`) VALUES (?, ?, ?, ?, ?)",(name,password,password1,email,number))
        con.commit()
        con.close()

        return render_template("signup-in.html")
    
    else:
        
        return render_template("signup-in.html")


@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('pass','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `name`, `password` from datas where `name` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signup-in.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup-in.html")


# ENDPOINTS
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/webcam')
def webcam():
    return(render_template('webcam.html'))


@app.route('/video_capture')
def video_capture():
    return(Response(usingWebcam(), mimetype='multipart/x-mixed-replace; boundary=frame'))

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'upload', secure_filename(f.filename))
        f.save(file_path)
        predictions = usingImage(file_path)
        return predictions
    return None


# MAIN
if __name__ == "__main__":
    app.run(debug=True, port=5001)