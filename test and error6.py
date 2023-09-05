import tkinter

import cv2
from colorama import Fore, Back, Style
import numpy as np
import cx_Oracle
con = cx_Oracle.connect("SYSTEM", "1234", "localhost/xe")
# import filedialog module
from tkinter import filedialog
# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("the_new_video_is.avi", fourcc , 25, (852, 480))

# repalce the test.mp4 with an video of your own 
camera = cv2.VideoCapture("bridge.mp4")

while True:
    _,img = camera.read()
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    count=0
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if(label=="car" or label=="truck" or label == "Motorcyle"):
                count=count+1
                a=count
                cursor = con.cursor()            
                sql_query = "INSERT INTO TABLE1 (SNO,LABEL,TYPE, COUNT) VALUES (" + str(a) + ",'" + label + "', 'HEAVY', "+str(count)+")"
               # print(sql_query)
                cursor.execute(sql_query)
                con.commit()
                cursor.close()
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x+17, y-5 ), font, 0.5, color, 1)
        
            
    color=colors[7]    
    cv2.putText(img, "count of vechiles : "+str(count) ,(50,50), font,0.5, color)        
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()

