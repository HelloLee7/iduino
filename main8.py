# 필요한 라이브러리 임포트
from urllib.request import urlopen  # URL을 열기 위한 라이브러리
import cv2  # OpenCV 라이브러리, 이미지 처리에 사용
import numpy as np  # NumPy 라이브러리, 배열 및 행렬 연산에 사용
import threading  # 스레딩 라이브러리, 병렬 처리를 위해 사용

# Keras 모델 로드
from keras.models import load_model  # Keras 모델을 로드하기 위한 함수
model = load_model("keras_model.h5", compile=False)  # 사전 학습된 Keras 모델을 로드

# 클래스 이름 로드
class_names = open("labels.txt", "r").readlines()  # 클래스 이름을 저장한 텍스트 파일을 읽어 리스트로 저장

# IP 주소 설정
ip = '192.168.137.152'  # 스트리밍 데이터를 가져올 IP 주소

# 스트리밍 데이터를 가져오기 위해 URL 열기
stream = urlopen('http://' + ip + ':81/stream')  # 스트리밍 데이터를 가져오기 위해 URL을 열고 연결

# 버퍼 초기화
buffer = b''  # 스트리밍 데이터를 저장할 버퍼를 빈 바이트 문자열로 초기화

# 속도를 100으로 설정하는 명령을 보내기
urlopen('http://' + ip + '/action?go=speed100')  # 차량의 속도를 100으로 설정하는 명령을 전송

# 이미지 처리 플래그 초기화
image_flag = 0  # 이미지 처리가 필요함을 나타내는 플래그를 0으로 초기화

# 이미지 처리 스레드 함수 정의
def image_process_thread():
    global img  # 전역 변수 img 사용
    global image_flag  # 전역 변수 image_flag 사용
    while True:  # 무한 루프
        if image_flag == 1:  # 이미지 처리가 필요할 때
            # 이미지를 Keras 모델 입력 형식에 맞게 변환
            img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)  # 이미지를 NumPy 배열로 변환하고 모델 입력 형식에 맞게 재구성
            img = (img / 127.5) - 1  # 이미지 정규화

            # 모델 예측 수행
            prediction = model.predict(img)  # 모델을 사용하여 예측 수행
            index = np.argmax(prediction)  # 가장 높은 확률을 가진 클래스의 인덱스 찾기
            class_name = class_names[index]  # 해당 인덱스의 클래스 이름 가져오기
            confidence_score = prediction[0][index]  # 해당 클래스의 확률 점수 가져오기
            percent = int(str(np.round(confidence_score * 100))[:2])  # 확률 점수를 백분율로 변환

            # 예측 결과에 따라 차량 제어 명령 전송
            if "go" in class_name[2:] and percent >= 95:  # 클래스 이름에 "go"가 포함되고 확률이 95% 이상일 때
                print("직진: ", str(np.round(confidence_score * 100))[:-2])  # 직진 명령 출력
                urlopen('http://' + ip + "/action?go=forward")  # 차량에 직진 명령 전송
            elif "left" in class_name[2:] and percent >= 95:  # 클래스 이름에 "left"가 포함되고 확률이 95% 이상일 때
                print("<: ", str(np.round(confidence_score * 100))[:-2])  # 좌회전 명령 출력
                urlopen('http://' + ip + "/action?go=left")  # 차량에 좌회전 명령 전송
            elif "right" in class_name[2:] and percent >= 95:  # 클래스 이름에 "right"가 포함되고 확률이 95% 이상일 때
                print(">: ", str(np.round(confidence_score * 100))[:-2])  # 우회전 명령 출력
                urlopen('http://' + ip + "/action?go=right")  # 차량에 우회전 명령 전송

            # 이미지 처리 플래그 초기화
            image_flag = 0  # 이미지 처리가 완료되었음을 나타내기 위해 플래그를 0으로 초기화
# 이미지 처리 스레드 시작
daemon_thread = threading.Thread(target=image_process_thread)
daemon_thread.daemon = True
daemon_thread.start()

# 무한 루프 시작
while True:
    # 스트리밍 데이터 읽기
    buffer += stream.read(4096)
    
    # JPEG 이미지의 시작과 끝을 찾기
    head = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')

    try:
        # JPEG 이미지가 버퍼에 존재하는지 확인
        if head > -1 and end > -1:
            # JPEG 이미지 데이터를 추출
            jpg = buffer[head:end+2]
            # 추출된 데이터를 버퍼에서 제거
            buffer = buffer[end+2:]
            # JPEG 데이터를 이미지로 디코딩
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            # 디코딩된 이미지를 화면에 표시
            cv2.imshow("AI CAR STREAMING", img)

            # 이미지의 높이와 너비를 가져오기
            height, width, _ = img.shape
            # 이미지의 하단 절반을 선택
            img = img[height // 2:, :]

            # 색상 범위를 설정하여 마스크 생성
            lower_bound = np.array([0, 0, 0])
            upper_bound = np.array([255, 255, 80])
            mask = cv2.inRange(img, lower_bound, upper_bound)

            # 생성된 마스크를 화면에 표시
            cv2.imshow("mask", mask)

            # 윤곽선 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # 윤곽선의 중심점 계산
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                # 중심점 오프셋 계산 (이미지의 중심과 객체의 중심 간의 거리)
                center_offset = width // 2 - cX

                # 객체의 중심에 원 그리기
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

            # 키 입력을 대기
            key = cv2.waitKey(1)
            # 'q' 키가 눌리면 루프를 종료하고 차량을 정지
            if key == ord('q'):
                urlopen('http://' + ip + "/action?go=stop")
                break

    except:
        # 오류가 발생하면 메시지를 출력
        print("error")
        pass

# 프로그램 종료 시 차량을 정지
urlopen('http://' + ip + "/action?go=stop")
# 모든 OpenCV 창을 닫기
cv2.destroyAllWindows()

# inRange 이진 마스크 생성
# findContours 이진 이미지에서 윤곽선을 찾아내고, 객체의 모양을 분석