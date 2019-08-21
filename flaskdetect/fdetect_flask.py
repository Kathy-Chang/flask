from flask_opencv_streamer.streamer import Streamer
from flask import Flask,render_template,Response
import cv2
import imutils
from imutils.video import VideoStream
import time
import paho.mqtt.client as mqtt
import sys
import os
import numpy as np
import argparse
import pickle
import webbrowser
import mysql.connector
import pymysql


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=False,default='haarcascade_frontalface_default.xml',
    help = "path to where the face cascade resides")
ap.add_argument("-d", "--detector", required=False,default='face_detection_model',
    help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=False,default='openface_nn4.small2.v1.t7',
    help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=False,default='output/recognizer.pickle',
    help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=False,default='output/le.pickle',
    help="path to label encoder")
ap.add_argument("-cc", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#mqtt subscribe
_g_cst_ToMQTTTopicServerIP = "192.168.21.128"
_g_cst_ToMQTTTopicServerPort = 1883 #port
_g_cst_MQTTTopicName = "WebToRpi" #TOPIC name

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("WebToRpi",0)
    

# The callback for when a PUBLISH message is received from the server.


def on_message(client, userdata, msg):
    global message
    msg.payload = msg.payload.decode("utf-8")
    print(msg.topic+" "+msg.payload)
    message = msg.payload
    
    
def on_connect2(client, userdata, flags, rc):
    print("Connected with result code c1: " + str(rc))
    client.subscribe("server1/test")

def on_message2(client, userdata, msg):
    print(msg.topic + ":" + str(msg.payload, encoding="utf-8"))    
    
    

message =""
mqttc = mqtt.Client("")
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.connect(_g_cst_ToMQTTTopicServerIP, _g_cst_ToMQTTTopicServerPort,60)
mqttc.subscribe("WebToRpi",0)
mqttc.loop_start()

detected = False
   
# app = Flask(__name__)
# @app.route("/")

mqttc2 = mqtt.Client("")
mqttc2.on_connect = on_connect2
mqttc2.on_message = on_message2
mqttc2.connect("localhost", _g_cst_ToMQTTTopicServerPort,60)

mqttc2.loop_start()





def detect_recognize_face():
    global detected,orig
    
    try:
        # initialize the video stream, allow the cammera sensor to warmup,
        # and initialize the FPS counter
        print("[INFO] starting video stream...")
        #vs = VideoStream(src=0).start()
        vs = VideoStream(usePiCamera=True).start()
        time.sleep(2.0)
        
        #while True:
        while True:
            print(message)
            # grab the frame from the threaded video stream
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            orig = frame.copy()
            #frame = imutils.resize(frame, width=100)

          
            # convert the test image to gray image as opencv face detector expects gray images 
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # load cascade classifier training file for haarcascade
            haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            haar_face_cascade.load('/home/pi/AIOT_Project/website/flaskdetect/haarcascade_frontalface_default.xml')

            # let's detect multiscale (some images may be closer to camera than others) images 
            faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);  

            # print the number of faces found
            print('Faces found: ', len(faces))

            # go over list of faces and draw them as rectangles on original colored
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            streamer.update_frame(frame)
            
            if not streamer.is_streaming:
                streamer.start_streaming()
            
            if message=="takepic":
                face_recognize()
                if detected == True:
                    break
            key = cv2.waitKey(1)
    finally:
        print("[INFO] quitting...")
        # do a bit of cleanup
        #vs.stop()
        mqttc.loop_stop()
        mqttc.disconnect()
        mqttc2.loop_stop()
        mqttc2.disconnect()


def face_recognize():
    global detected, orig, name
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(args["recognizer"], "rb").read())
    le = pickle.loads(open(args["le"], "rb").read())

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    #image = cv2.imread(image)
    image = imutils.resize(orig, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]+"@aiot.com"

            # draw the bounding box of the face along with the associated
            # probability
            #text = "{:.2f}".format(proba * 100)
            text = int(proba * 100)
            if text>=70 and name != "unknown@aiot.com":
                print("open "+name)
                detected = True
                mydb()
                break
            else:
                mqttc2.publish("RpiToWeb","no")
  
                print("No one is detected!!")
                break
            # Part of drwaing the bounding box
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(image, (startX, startY), (endX, endY),
                #(0, 0, 255), 2)
            # cv2.putText(image, text, (startX, y),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show the output image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

def mydb():
    global name
    
    print("here is db "+name)
    
    mydb = mysql.connector.connect(host="192.168.21.140",user="root", passwd="12345678", database="stkdb")
    cur = mydb.cursor()
    #sql =("""INSERT INTO eardata(datetime,earRecord) VALUES(%s, %s)""",(secs, earRecord))
    sql =("SELECT emailaddress from login WHERE emailaddress=\"%s\"",name)
    #sql =("SELECT emailaddress from login WHERE emailaddress="+"'"+name+"'")
    print(sql)
    
    try:
        print("Writing to the database...")
        
        cur.execute(*sql)
        
        
        #mydb.commit()
        
        #myresult = cur.fetchall()
        myresult = cur.fetchone()
        
        if myresult != "":
            print("Login")
            mqttc2.publish("RpiToWeb","yes")
        else:
            mqttc2.publish("RpiToWeb","no")
        #for x in myresult:
        #    print(x)
        
        
        print ("Write complete")
        
        #mydb.commit()
        cur.close()
        mydb.close()
        
        
    #except:
        #mydb.rollback()
        #print("We have a problem")
        
    except Exception: #方法一：捕获所有异常  
        #如果发生异常，则回滚  
        print("发生异常",Exception)  
        mydb.rollback() 
        
        

    

while True:
    if message == "go":
        
        port = 3030
        require_login = False
        stream_res=(400, 300)
        streamer = Streamer(port, require_login, stream_res)
        detect_recognize_face()
        break




    



