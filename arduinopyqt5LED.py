import serial
import time

try:
    # 아두이노 시리얼 통신 설정
    arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
    time.sleep(2)  # 아두이노 리셋 대기
    
    def send_command(command):
        arduino.write(command.encode('utf-8'))
        arduino.flush()  # 버퍼 비우기
        time.sleep(0.1)  # 명령 처리 대기
    
    # LED 상태 토글
    while True:
        send_command('l')  # LED 켜기
        print("LED ON")
        time.sleep(2)      # 2초 대기
        send_command('0')  # LED 끄기
        print("LED OFF")
        time.sleep(2)      # 2초 대기

except serial.SerialException as e:
    print(f"시리얼 포트 연결 실패: {e}")
finally:
    if 'arduino' in locals():
        arduino.close()