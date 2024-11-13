import torch
import cv2
from numpy import random
import numpy as np
from urllib.request import urlopen
import threading
import time
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

ip = '192.168.137.74'
stream = urlopen('http://' + ip +':81/stream')
buffer = b''
urlopen('http://' + ip + "/action?go=speed40")

# YOLOv5 모델 정의
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

yolo_state = "go"
thread_frame = None  
image_flag = 0
thread_image_flag = 0
def yolo_thread():
    global image_flag,thread_image_flag,frame, thread_frame, yolo_state
    while True:
        if image_flag == 1:
            thread_frame = frame
            
            # 이미지를 모델에 입력
            results = model(thread_frame)

            # 객체 감지 결과 얻기
            detections = results.pandas().xyxy[0]

            if not detections.empty:
                # 결과를 반복하며 객체 표시
                for _, detection in detections.iterrows():
                    x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values
                    label = detection['name']
                    conf = detection['confidence']

                    if "stop" in label and conf > 0.3:
                        print("stop")
                        yolo_state = "stop"
                    # elif "slow" in label and conf > 0.3:
                    #     print("slow")
                    #     yolo_state = "go"
                    #     urlopen('http://' + ip + "/action?go=speed40")
                    # elif "speed50" in label and conf > 0.3:
                    #     print("speed50")
                    #     yolo_state = "go"
                    #     urlopen('http://' + ip + "/action?go=speed60")

                    # 박스와 라벨 표시
                    color = [int(c) for c in random.choice(range(256), size=3)]
                    cv2.rectangle(thread_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(thread_frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            thread_image_flag = 1
            image_flag = 0
            
# 데몬 스레드를 생성합니다.
t1 = threading.Thread(target=yolo_thread)
t1.daemon = True 
t1.start()


def image_process_thread():
    global image_flag, car_state, yolo_state
    while True:
        if image_flag == 1:
            if car_state == "go" and yolo_state =="go":
                urlopen('http://' + ip + "/action?go=forward")
            elif car_state == "right" and yolo_state =="go":
                urlopen('http://' + ip + "/action?go=right")
            elif car_state == "left" and yolo_state =="go":
                urlopen('http://' + ip + "/action?go=left")
            elif yolo_state =="stop":
                urlopen('http://' + ip + "/action?go=stop")
            
            image_flag = 0
            
# 데몬 스레드를 생성합니다.
t2 = threading.Thread(target=image_process_thread)
t2.daemon = True 
t2.start()

while True:
    buffer += stream.read(4096)
    head = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')
    
    try: 
        if head > -1 and end > -1:
            jpg = buffer[head:end+2]
            buffer = buffer[end+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            img = cv2.flip(img, -1)
            # 프레임 크기 조정
            frame = cv2.resize(img, (640, 480))

            height, width, _ = img.shape
            img = img[height // 2:, :]
            
            # 색상 필터링으로 검정색 선 추출
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([0, 0, 0])
            upper_bound = np.array([255, 255, 80])
            mask = cv2.inRange(img, lower_bound, upper_bound)
            
            # 무게 중심 계산
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            # 무게 중심과 이미지 중앙의 거리 계산
            center_offset = width // 2 - cX
            #print(center_offset)

            # 디버그용 시각화
            cv2.circle(img, (cX, cY), 10, (0, 255, 0), -1)
            cv2.imshow("AI CAR Streaming", img)

            if center_offset > 10:
                print("오른쪽")
                car_state = "right"
            elif center_offset < -10:
                print("왼쪽")
                car_state = "left"
            else:
                print("직진")
                car_state = "go"

            image_flag = 1

            #쓰레드에서 이미지 처리가 완료되었으면
            if thread_image_flag == 1:
                cv2.imshow('thread_frame', thread_frame)
                thread_image_flag = 0

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    except:
        print("에러")
        pass

urlopen('http://' + ip + "/action?go=stop")
cv2.destroyAllWindows()

# main6-3-2.py
# 자율주행에 객체 검출 결과 반영하기
