import serial
import time

# 시리얼 포트 설정 (포트 이름은 시스템에 따라 다를 수 있습니다)
ser = serial.Serial('COM4', 9600)  # Windows의 경우 COM 포트 사용
time.sleep(0.5)  # 아두이노 초기화 시간 대기

# LED 켜기
ser.write(b'1')
time.sleep(0.5)  # 2초 대기

# LED 끄기
ser.write(b'0')
time.sleep(0.5)  # 2초 대기

ser.close()