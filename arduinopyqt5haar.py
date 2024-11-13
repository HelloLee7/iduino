import torch
import torchvision
import cv2
from numpy import random
import numpy as np
from urllib.request import urlopen
import threading
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Haar Cascade 파일 경로
cascade_path = "haarcascade_frontalface_default.xml"

# Haar Cascade 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

# IP 카메라 스트림 설정
ip = '192.168.137.74'
stream = urlopen('http://' + ip + ':81/stream')
buffer = b''
urlopen('http://' + ip + "/action?go=speed40")

# 얼굴 검출 활성화 여부
detect_faces = True

while True:
    # 스트림에서 프레임 읽기
    buffer += stream.read(1024)
    a = buffer.find(b'\xff\xd8')
    b = buffer.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = buffer[a:b+2]
        buffer = buffer[b+2:]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, -1)
        # 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if detect_faces:
            # 얼굴 검출
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # 검출된 얼굴에 사각형 그리기
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 결과 프레임 표시
        cv2.imshow('Face Detection', frame)
        
        # 키보드 입력 감지
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u'):
            detect_faces = not detect_faces  # 'u' 버튼을 누르면 얼굴 검출 활성화/비활성화 전환

# 캡처 객체와 윈도우 해제
cv2.destroyAllWindows()