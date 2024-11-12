import sys  # 시스템 관련 기능을 사용하기 위해 임포트
import threading  # 스레드를 사용하기 위해 임포트
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QDialog  # PyQt5에서 제공하는 위젯 클래스 임포트
from PyQt5.QtCore import Qt, QTimer  # 정렬 옵션을 사용하기 위해 임포트
from PyQt5.QtGui import QPixmap, QImage  # 이미지를 표시하기 위해 임포트
import cv2  # OpenCV를 사용하여 이미지 처리를 위해 임포트
import numpy as np  # NumPy를 사용하여 배열 처리를 위해 임포트
from urllib.request import urlopen  # URL을 열기 위해 임포트
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import subprocess
import torch
import torchvision
# 전역 변수
thread_frame = None

class YOLOWindow(QDialog):  # QDialog를 상속받아 YOLOWindow 클래스 정의
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Window")  # 새로운 창의 제목 설정
        self.setGeometry(100, 100, 640, 480)  # 창의 위치와 크기 설정

        # 레이아웃 생성
        layout = QVBoxLayout()

        # QLabel 위젯 생성 및 설정
        self.label = QLabel("This is the YOLO window")
        self.label.setAlignment(Qt.AlignCenter)  # 텍스트를 중앙에 정렬
        layout.addWidget(self.label)  # QLabel을 레이아웃에 추가

        self.setLayout(layout)  # 레이아웃을 창에 설정

        # 타이머 설정
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(30)  # 30ms마다 업데이트

    def update_image(self):
        global thread_frame
        if thread_frame is not None:
            # OpenCV 이미지를 QImage로 변환
            height, width, channel = thread_frame.shape
            bytes_per_line = 3 * width #픽셀 당 3바이트를 사용하므로 3 * width로 계산
            q_img = QImage(thread_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.label.setPixmap(QPixmap.fromImage(q_img))  

class MainWindow(QMainWindow):  # QMainWindow를 상속받아 MainWindow 클래스 정의
    def __init__(self):  # 클래스 초기화 메서드
        super().__init__()  # 부모 클래스의 초기화 메서드 호출

        self.setWindowTitle("Arduino Test")  # 윈도우 제목 설정
        self.setFocusPolicy(Qt.StrongFocus)  # 키 이벤트를 받기 위해 포커스 정책 설정
        # 메인 레이아웃 생성
        main_layout = QVBoxLayout()

        # QLabel 위젯 생성 및 설정
        label = QLabel("Arduino Test")  # "Arduino Test" 텍스트를 가진 QLabel 생성
        label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)  # 텍스트를 상단 중앙에 정렬
        label.setFixedHeight(30)  # QLabel의 높이를 30으로 설정
        main_layout.addWidget(label)  # QLabel을 레이아웃의 첫 번째 위젯으로 추가


        new_button_layout = QHBoxLayout()
        new_button = QPushButton("haal")  # "New Button" 텍스트를 가진 QPushButton 생성
        new_button.clicked.connect(self.on_new_button_clicked)  # 버튼 클릭 시 on_new_button_clicked 메서드 호출
        new_button_layout.addWidget(new_button)  # 버튼을 레이아웃에 추가
        main_layout.addLayout(new_button_layout)  # 메인 레이아웃에 새로운 버튼 레이아웃 추가

        # Open YOLO 버튼 레이아웃 생성
        openyolo_layout = QHBoxLayout()
        yolos = ["Open YOLO"]  # 제어 버튼에 사용할 명령 리스트
        for yolo in yolos:
            button = QPushButton(yolo)  # "Open YOLO" 텍스트를 가진 QPushButton 생성
            button.clicked.connect(self.on_open_yolo_button_clicked)  # 버튼 클릭 시 on_open_yolo_button_clicked 메서드 호출
            openyolo_layout.addWidget(button)  # 버튼을 레이아웃에 추가
        main_layout.addLayout(openyolo_layout)  # 레이아웃을 메인 레이아웃에 추가

        # 속도 버튼 레이아웃 생성
        speed_layout = QHBoxLayout()

        # 속도 버튼 생성 및 설정
        speeds = [40, 50, 60, 80, 100]  # 속도 버튼에 사용할 속도 값 리스트
        for speed in speeds:  # 각 속도 값에 대해 반복
            button = QPushButton(f"speed: {speed}")  # 속도 값을 가진 QPushButton 생성
            button.clicked.connect(lambda checked, s=speed: self.on_speed_button_clicked(s))  # 버튼 클릭 시 on_speed_button_clicked 메서드 호출
            speed_layout.addWidget(button)  # 버튼을 속도 버튼 레이아웃에 추가

        # 메인 레이아웃에 속도 버튼 레이아웃 추가
        main_layout.addLayout(speed_layout)

        # 제어 버튼 레이아웃 생성
        control_layout = QHBoxLayout()

        # 제어 버튼 생성 및 설정
        controls = ["Turn left", "left", "forward", "backward", "stop", "right", "Turn right"]  # 제어 버튼에 사용할 명령 리스트
# 제어 버튼 생성 및 설정
        for control in controls:  # 각 명령에 대해 반복
            button = QPushButton(control)  # 명령을 가진 QPushButton 생성
            button.pressed.connect(lambda c=control: self.on_control_button_pressed(c))  # 버튼 누를 때 호출
            button.released.connect(lambda: self.on_control_button_released())  # 버튼 뗄 때 호출
            control_layout.addWidget(button)  # 버튼을 제어 버튼 레이아웃에 추가
        main_layout.addLayout(control_layout)  # 메인 레이아웃에 제어 버튼 레이아웃 추가


        # 중앙 위젯 설정
        container = QWidget()  # 중앙 위젯 생성
        container.setLayout(main_layout)  # 중앙 위젯에 메인 레이아웃 설정
        self.setCentralWidget(container)  # 중앙 위젯을 메인 윈도우의 중앙 위젯으로 설정

        # 스트리밍 데이터 읽기 스레드 시작
        # self.streaming_thread = threading.Thread(target=self.read_stream)  # read_stream 메서드를 실행하는 스레드 생성
        # self.streaming_thread.daemon = True  # 스레드를 데몬 스레드로 설정
        # self.streaming_thread.start()  # 스레드 시작

    def on_open_yolo_button_clicked(self):  # Open YOLO 버튼 클릭 시 호출되는 메서드
        print("Open YOLO button clicked")  # 클릭된 버튼의 텍스트를 출력
        python_executable = sys.executable  # 현재 사용 중인 Python 실행 파일 경로
        subprocess.Popen([python_executable, "arduinopyqt5yolo.py"])  # 현재 Python 실행 파일을 사용하여 스크립트 실행

    def on_control_button_pressed(self, control):  # 버튼 누를 때 동작
        print(f"{control.capitalize()} button pressed")
        self.send_command_to_arduino(control) 

    def on_control_button_released(self):  # 버튼 뗄 때 정지
        print("Stop button released")
        self.send_command_to_arduino("stop")


    def on_speed_button_clicked(self, speed):  # 속도 버튼 클릭 시 호출되는 메서드
        print(f"Speed {speed} button clicked")  # 클릭된 버튼의 속도를 출력
        self.send_command_to_arduino(speed)        # 여기에 시리얼 포트를 통해 명령 전송 코드를 추가할 수 있습니다

    def on_new_button_clicked(self):
        print("New Button clicked")
        python_executable = sys.executable  # 현재 사용 중인 Python 실행 파일 경로
        subprocess.Popen([python_executable, "arduinopyqt5haar.py"])         
        # 새로운 버튼 클릭 시 수행할 동작을 여기에 추가하세요
# 제어 명령을 Arduino로 전송하는 메서드 호출

    def send_command_to_arduino(self, command):  # 제어 명령을 Arduino로 전송하는 메서드
        ip = '192.168.137.84'  # Arduino의 IP 주소
        if command == "forward":  # 명령이 "forward"인 경우
            print('전진')  # "전진" 출력
            urlopen('http://' + ip + "/action?go=forward")  # Arduino로 전진 명령 전송
        elif command == "left":  # 명령이 "left"인 경우
            print('왼쪽')  # "왼쪽" 출력
            urlopen('http://' + ip + "/action?go=left")  # Arduino로 왼쪽 명령 전송
        elif command == "right":  # 명령이 "right"인 경우
            print('오른쪽')  # "오른쪽" 출력
            urlopen('http://' + ip + "/action?go=right")  # Arduino로 오른쪽 명령 전송
        elif command == "backward":  # 명령이 "backward"인 경우
            print('후진')  # "후진" 출력
            urlopen('http://' + ip + "/action?go=backward")  # Arduino로 후진 명령 전송
        elif command == "Turn left":  # 명령이 "Turn left"인 경우
            print('왼쪽 회전')  # "왼쪽 회전" 출력
            urlopen('http://' + ip + "/action?go=turn_left")  # Arduino로 왼쪽 회전 명령 전송
        elif command == "Turn right":  # 명령이 "Turn right"인 경우
            print('오른쪽 회전')  # "오른쪽 회전" 출력
            urlopen('http://' + ip + "/action?go=turn_right")  # Arduino로 오른쪽 회전 명령 전송
        elif command == "stop":  # 명령이 "stop"인 경우
            print('정지')  # "정지" 출력
            urlopen('http://' + ip + "/action?go=stop")  # Arduino로 정지 명령 전송

        elif command == "speed: 40":  #  
            print('40')   
            urlopen('http://' + ip + "/action?go=40") 
        elif command == "speed: 50":  #  
            print('50')   
            urlopen('http://' + ip + "/action?go=50") 
        elif command == "speed: 60":  #  
            print('60')   # 올바른 출력 메시지로 수정
            urlopen('http://' + ip + "/action?go=60") 
        elif command == "speed: 80":  #  
            print('80')   
            urlopen('http://' + ip + "/action?go=80") 
        elif command == "speed: 100":  #  
            print('100')   
            urlopen('http://' + ip + "/action?go=100")        
                    
    # def read_stream(self):  # 스트리밍 데이터를 읽는 메서드
    #     global thread_frame
    #     ip = '192.168.137.89'  # Arduino의 IP 주소
    #     stream = urlopen('http://' + ip + ':81/stream')  # 스트리밍 데이터를 가져오기 위해 URL 열기
    #     buffer = b''  # 스트리밍 데이터를 저장할 버퍼 초기화

    #     urlopen('http://' + ip + "/action?go=speed40")  # 초기 속도를 40으로 설정

    #     while True:  # 무한 루프
    #         buffer += stream.read(4096)  # 스트리밍 데이터 읽기
    #         head = buffer.find(b'\xff\xd8')  # JPEG 이미지의 시작을 찾기
    #         end = buffer.find(b'\xff\xd9')  # JPEG 이미지의 끝을 찾기

    #         try:
    #             if head > -1 and end > -1:  # JPEG 이미지가 버퍼에 존재하는지 확인
    #                 jpg = buffer[head:end+2]  # JPEG 이미지 데이터를 추출
    #                 buffer = buffer[end+2:]  # 추출된 데이터를 버퍼에서 제거
    #                 img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)  # JPEG 데이터를 이미지로 디코딩

    #                 # 아래부분의 반만 자르기
    #                 height, width, _ = img.shape  # 이미지의 높이와 너비 가져오기
    #                 img = img[height // 2:, :]  # 이미지의 아래 부분 절반 자르기
    #         except Exception as e:  # 예외 발생 시
    #             print(f"Error: {e}")  # 오류 메시지 출력

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_W:
            print('전진')
            self.send_command_to_arduino('forward')
        elif event.key() == Qt.Key_A:
            print('왼쪽')
            self.send_command_to_arduino('left')
        elif event.key() == Qt.Key_D:
            print('오른쪽')
            self.send_command_to_arduino('right')
        elif event.key() == Qt.Key_S:
            print('후진')
            self.send_command_to_arduino('backward')
        elif event.key() == Qt.Key_X:
            print('정지')
            self.send_command_to_arduino('stop')
        else:
            super().keyPressEvent(event)



if __name__ == "__main__":  # 메인 함수
    app = QApplication(sys.argv)  # 애플리케이션 객체 생성

    window = MainWindow()  # 메인 윈도우 객체 생성
    window.resize(600, 400)  # 윈도우 크기 설정
    window.show()  # 윈도우 표시

    sys.exit(app.exec_())  # 애플리케이션 이벤트 루프 실행