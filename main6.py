from urllib.request import urlopen
import cv2
import numpy as np
import os
from keras.models import load_model
import threading
import time

model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt","r").readlines()

ip = '192.168.137.15'
stream = urlopen('http://' + ip +':81/stream')
buffer = b''
urlopen('http://'+ip +'/action?go=speed100')

image_flag = 0

def image_process_thread():
    global img
    global image_flag
    while True:
        if image_flag == 1:
            img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
            img = (img/127.5) - 1

            prediction = model.predict(img)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            percent = int(str(np.round(confidence_score * 100))[:2])

            if "go"in class_name[2:] and percent >=95:
                print("직진: ",str(np.round(confidence_score * 100))[:-2])
                urlopen('http://'+ip+"/action?go=forward")
            elif "left"in class_name[2:] and percent >=95:
                print("<: ",str(np.round(confidence_score * 100))[:-2])
                urlopen('http://'+ip+"/action?go=left")
            elif "right"in class_name[2:] and percent >=95:
                print(">: ",str(np.round(confidence_score * 100))[:-2])
                urlopen('http://'+ip+"/action?go=right")

            image_flag=0

daemon_thread = threading.Thread(target=image_process_thread)
daemon_thread.daemon = True
daemon_thread.start()

while True:
    buffer += stream.read(4096)
    head = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')

    try:
        if head >-1 and end >-1:
            jpg = buffer[head:end+2]
            buffer = buffer[end+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            height, width, _ = img.shape
            img = img[height //2:, :]

            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
            cv2.imshow("AI CAR STREAMING", img)
            image_flag = 1

            key = cv2.waitKey(1)
            if key == ord('q'):
                urlopen('http://' + ip +"/action?go=stop")
                break
                                                                                
    except:
        print("error")
        pass

urlopen('http://'+ip+"/action?go=stop")
cv2.destroyAllWindows()

#inRange 이진 마스크 생성
#findContours 이진 이미지에서 윤곽선을 찾아내고, 객체의 모양을 분석