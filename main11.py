import torch  # PyTorch 라이브러리, 딥러닝 모델을 로드하고 실행하는 데 사용
import cv2  # OpenCV 라이브러리, 이미지 및 비디오 처리에 사용
from numpy import random  # NumPy 라이브러리의 random 모듈, 랜덤 숫자 생성에 사용
import numpy as np  # NumPy 라이브러리, 배열 및 행렬 연산에 사용
from urllib.request import urlopen  # URL을 열기 위한 라이브러리

# IP 주소 설정 (빈 문자열로 초기화, 실제 IP 주소로 변경 필요)
ip = ''

# 스트리밍 데이터를 가져오기 위해 URL 열기
stream = urlopen('http://' + ip + ':81/stream')  # 스트리밍 데이터를 가져오기 위해 URL을 열고 연결
buffer = b''  # 스트리밍 데이터를 저장할 버퍼를 빈 바이트 문자열로 초기화

# 속도를 40으로 설정하는 명령을 보내기
urlopen('http://' + ip + '/action?go=speed40')  # 차량의 속도를 40으로 설정하는 명령을 전송

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 사전 학습된 YOLOv5 모델을 로드

# CUDA가 사용 가능한 경우 모델을 GPU로 이동
if torch.cuda.is_available():
    model = model.cuda()

# 무한 루프 시작
while True:
    buffer += stream.read(4096)  # 스트리밍 데이터 읽기
    head = buffer.find(b'\xff\xd8')  # JPEG 이미지의 시작을 찾기
    end = buffer.find(b'\xff\xd9')  # JPEG 이미지의 끝을 찾기

    try:
        if head > -1 and end > -1:  # JPEG 이미지가 버퍼에 존재하는지 확인
            jpg = buffer[head:end+2]  # JPEG 이미지 데이터를 추출
            buffer = buffer[end+2:]  # 추출된 데이터를 버퍼에서 제거
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)  # JPEG 데이터를 이미지로 디코딩
            frame = cv2.resize(img, (640, 480))  # 이미지를 640x480 크기로 조정
            results = model(frame)  # 모델을 사용하여 이미지에서 객체 감지 수행

            detections = results.pandas().xyxy[0]  # 감지된 객체의 좌표 및 정보를 Pandas DataFrame으로 가져오기

            if not detections.empty:  # 감지된 객체가 있을 경우
                for _, detection in detections.iterrows():  # 각 감지된 객체에 대해 반복
                    x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values  # 객체의 좌표 가져오기
                    label = detection['name']  # 객체의 레이블 가져오기
                    conf = detection['confidence']  # 객체의 신뢰도 가져오기

                    color = [int(c) for c in np.random.choice(range(256), size=3)]  # 랜덤 색상 생성
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 객체 주위에 사각형 그리기
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 객체 레이블 및 신뢰도 표시

        cv2.imshow('frame', frame)  # 프레임을 화면에 표시

        key = cv2.waitKey(1)  # 키 입력 대기
        if key == ord('q'):  # 'q' 키를 누르면 루프 종료
            break

    except:  # 예외 발생 시
        print('error')  # 오류 메시지 출력
        pass

cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기