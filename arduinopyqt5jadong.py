import cv2
import numpy as np
from urllib.request import urlopen
import threading

ip = '192.168.137.74'
stream = urlopen('http://' + ip + ':81/stream')
buffer = b''
urlopen('http://' + ip + "/action?go=speed40")

image_flag = 0
def image_process_thread():
    global image_flag, car_state
    while True:
        if image_flag == 1:
            if car_state == "go":
                urlopen('http://' + ip + "/action?go=forward")
            elif car_state == "right":
                urlopen('http://' + ip + "/action?go=right")
            elif car_state == "left":
                urlopen('http://' + ip + "/action?go=left")
            
            image_flag = 0
            
# 데몬 스레드를 생성합니다.
daemon_thread = threading.Thread(target=image_process_thread)
daemon_thread.daemon = True 
daemon_thread.start()

car_state = "go"
while True:
    buffer += stream.read(4096)
    head = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')
    
    try:
        if head > -1 and end > -1:
            jpg = buffer[head:end+2]
            buffer = buffer[end+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            #cv2.imshow("AI CAR Streaming", img)
            
            height, width, _ = img.shape
            img = img[height // 2:, :]
            
            # 색상 필터링으로 검정색 선 추출
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([0, 0, 0])
            upper_bound = np.array([255, 255, 80])
            mask = cv2.inRange(img, lower_bound, upper_bound)

            cv2.imshow("mask", mask)
            
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
                #urlopen('http://' + ip + "/action?go=right")
            elif center_offset < -10:
                print("왼쪽")
                car_state = "left"
                #urlopen('http://' + ip + "/action?go=left")
            else:
                print("직진")
                car_state = "go"
                #urlopen('http://' + ip + "/action?go=forward")

            image_flag = 1
            key = cv2.waitKey(1)
            if key == ord('q'):
                urlopen('http://' + ip + "/action?go=stop")
                break

    except:
        print("에러")
        pass

urlopen('http://' + ip + "/action?go=stop")
cv2.destroyAllWindows()

# main5-2-2.py
# 쓰레드를 이용하여 자율주행 성능 높히기