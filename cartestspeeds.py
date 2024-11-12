def send_command_to_arduino(self, command):
    ip = '192.168.137.84'  # Arduino의 IP 주소
    command = str(command)  # 전달된 command 값을 문자열로 변환
    
    if command == "forward":  
        print('전진')  
        urlopen('http://' + ip + "/action?go=forward")  
    elif command == "left":  
        print('왼쪽')  
        urlopen('http://' + ip + "/action?go=left")  
    elif command == "right":  
        print('오른쪽')  
        urlopen('http://' + ip + "/action?go=right")  
    elif command == "backward":  
        print('후진')  
        urlopen('http://' + ip + "/action?go=backward")  
    elif command == "Turn left":  
        print('왼쪽 회전')  
        urlopen('http://' + ip + "/action?go=turn_left")  
    elif command == "Turn right":  
        print('오른쪽 회전')  
        urlopen('http://' + ip + "/action?go=turn_right")  
    elif command == "stop":  
        print('정지')  
        urlopen('http://' + ip + "/action?go=stop")  
    elif command in ["40", "50", "60", "80", "100"]:  # 속도 값도 문자열로 처리
        print(f'속도 {command}')
        urlopen(f'http://{ip}/action?go={command}')
