# IP 주소 설정
ip = '192.168.137.50'

# 스트리밍 데이터를 가져오기 위해 URL 열기
stream = urlopen('http://' + ip + ':81/stream')

# 버퍼 초기화
buffer = b''

# 속도를 100으로 설정하는 명령을 보내기
urlopen('http://' + ip + '/action?go=speed100')

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
            cv2.imshow("img", img)

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