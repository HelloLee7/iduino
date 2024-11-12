import torch  # PyTorch 라이브러리, 딥러닝 모델을 로드하고 실행하는 데 사용
import cv2  # OpenCV 라이브러리, 이미지 및 비디오 처리에 사용
from numpy import random  # NumPy 라이브러리의 random 모듈, 랜덤 숫자 생성에 사용
import numpy as np  # NumPy 라이브러리, 배열 및 행렬 연산에 사용
from urllib.request import urlopen  # URL을 열기 위한 라이브러리
import threading  # 스레딩 라이브러리, 병렬 처리를 위해 사용
import time
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# IP 주소 설정 (빈 문자열로 초기화, 실제 IP 주소로 변경 필요)
ip = '192.168.137.116'

# 스트리밍 데이터를 가져오기 위해 URL 열기
stream = urlopen('http://' + ip + ':81/stream')  # 스트리밍 데이터를 가져오기 위해 URL을 열고 연결
buffer = b''  # 스트리밍 데이터를 저장할 버퍼를 빈 바이트 문자열로 초기화

# 속도를 40으로 설정하는 명령을 보내기
urlopen('http://' + ip + '/action?go=speed40')  # 차량의 속도를 40으로 설정하는 명령을 전송

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # 사전 학습된 YOLOv5 모델을 로드

yolo_state = 'go'
thread_frame = None
image_flag = 0  # 이미지 처리가 필요함을 나타내는 플래그를 0으로 초기화
thread_image_flag = 0  # 이미지 처리가 필요함을 나타내는 플래그를 0으로 초기화
def yolo_thread():
    global image_flag, thread_image_flag, frame, thread_frame, yolo_state
    while True:
        if image_flag == 1:
            thread_frame = frame
            results = model(thread_frame)
            detections = results.pandas().xyxy[0]  # 감지된 객체의 좌표 및 정보를 Pandas DataFrame으로 가져오기

        if not detections.empty:
            for _, detection in detections.iterrows():
                x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).val-ues  # 객체의 좌표 가져오기
                label = detection['name']  # 객체의 레이블 가져오기
                conf = detection['confidence']  # 객체의 신뢰도 가져오기

                if 'stop' in label and conf >= 0.3:
                    yolo_state = 'stop'
                    print('stop')
                elif 'slow' in label and conf >= 0.3:
                    yolo_state = 'go'
                    print('slow')
                    urlopen('http://' + ip + '/action?go=speed40')
                elif 'speed50' in label and conf >= 0.3:
                    yolo_state = 'go'
                    print('speed50')
                    urlopen('http://' + ip + '/action?go=speed60')  

                color = [int(c) for c in np.random.choice(range(256), size=3)]  # 랜덤 색상 생성
                cv2.rectangle(thread_frame, (x1, y1), (x2, y2), color, 2)  # 객체 주위에 사각형 그리기
                cv2.putText(thread_frame, f'{label} {conf:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 객체 레이블 및 신뢰도 표시
                thread_image_flag = 1
                image_flag = 0

t1 = threading.Thread(target=yolo_thread)
t1.daemon = True    
t1.start()

def image_process_thread():
    global image_flag, car_state, yolo_state
    while True:
        if image_flag == 1:
            if car_state == 'go'and yolo_state == 'go':
                urlopen('http://' + ip + '/action?go=forward')
            elif car_state == 'right'and yolo_state == 'go':
                urlopen('http://' + ip + '/action?go=right')
            elif car_state == 'left'and yolo_state == 'go':
                urlopen('http://' + ip + '/action?go=left')
            elif yolo_state == 'stop':
                urlopen('http://' + ip + '/action?go=stop')                                
            image_flag = 0

t2 = threading.Thread(target=image_process_thread)
t2.daemon = True    
t2.start()

while True:
    buffer += stream.read(4096)  # 스트리밍 데이터 읽기
    head = buffer.find(b'\xff\xd8')  # JPEG 이미지의 시작을 찾기
    end = buffer.find(b'\xff\xd9')  # JPEG 이미지의 끝을 찾기

    try:
        # JPEG 이미지가 버퍼에 존재하는지 확인
        if head > -1 and end > -1:
            # JPEG 이미지 데이터를 추출
            jpg = buffer[head:end+2]
            # 추출된 데이터를 버퍼에서 제거
            buffer = buffer[end+2:]
            # JPEG 데이터를 이미지로 디코딩
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


            frame = cv2.resize(img, (640, 480))

            height, width, _ = img.shape
            # 이미지의 하단 절반을 선택
            img = img[height // 2:, :]

            # 색상 범위를 설정하여 마스크 생성
            lower_bound = np.array([0, 0, 0])
            upper_bound = np.array([255, 255, 80])
            mask = cv2.inRange(img, lower_bound, upper_bound)

            M = cv2.moments(mask)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            center_offset = width // 2 - cX

            cv2.circle(img, (cX, cY), 10, (0, 255, 0), -1)
                # 이미지를 화면에 표시
            cv2.imshow("AI CAR STREAMING", img)

                # 중심점 오프셋에 따라 차량의 방향 결정
            if center_offset > 10:
                    # 중심점이 오른쪽에 있으면 오른쪽으로 이동
                print("오른쪽")
                car_state = "right"
            elif center_offset < -10:
                    # 중심점이 왼쪽에 있으면 왼쪽으로 이동
                print("왼쪽")
                car_state = "left"
            else:
                    # 중심점이 중앙에 가까우면 직진
                print("직진")
                car_state = "go"

                # 이미지 처리가 완료되었음을 표시
            image_flag = 1            

        if thread_image_flag == 1:
            cv2.imshow('frame', thread_frame)
            thread_image_flag = 0

        key = cv2.waitKey(1)  # 키 입력 대기
        if key == ord('q'):  # 'q' 키를 누르면 루프 종료
            break

    except:  # 예외 발생 시
        print('error')  # 오류 메시지 출력
        pass

urlopen('http://' + ip + '/action?go=stop')  # 차량의 속도를 0으로 설정하는 명령을 전송
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기