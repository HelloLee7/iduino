
import socket
from urllib.request import urlopen

def get_arduino_ip(arduino_hostname='arduino.local'):
    try:
        arduino_ip = socket.gethostbyname(arduino_hostname)
        print(f"Arduino IP 주소: {arduino_ip}")
        return arduino_ip
    except socket.error as e:
        print(f"Arduino IP 주소를 가져오는 데 실패했습니다: {e}")
        return None

def send_command_to_arduino(command, arduino_hostname='arduino.local'):
    arduino_ip = get_arduino_ip(arduino_hostname)
    if not arduino_ip:
        print("Arduino IP 주소를 찾을 수 없습니다.")
        return
    try:
        # 공백이 있는 명령은 URL 인코딩 처리
        encoded_command = command.replace(' ', '%20')
        urlopen(f'http://{arduino_ip}/action?go={encoded_command}')
        print(f"명령 '{command}'를 Arduino({arduino_ip})로 전송했습니다.")
    except Exception as e:
        print(f"명령 '{command}' 전송에 실패했습니다: {e}")