from urllib.request import urlopen
import cv2
import numpy as np

ip = '192.168.137.15'
stream = urlopen('http://' + ip +':81/stream')
buffer = b''

while True:
    buffer += stream.read(4096)
    head = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')

    try:
        if head >-1 and end >-1:
            jpg = buffer[head:end+2]
            buffer = buffer[end+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            height, width, _ = img.shape
            img = img[height //2:, :]

            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
            cv2.imshow("AI CAR STREAMING", img)

            key = cv2.waitKey(1)
            if key == ord('q'):
                urlopen('http://'+ ip +"/action?go=stop")
                break
            elif key == ord('w'):
                urlopen('http://'+ ip +"/action?go=forward")
                print("전진")
            elif key == ord('a'):
                urlopen('http://'+ ip +"/action?go=left")
                print("왼쪽")                
            elif key == ord('s'):
                urlopen('http://'+ ip +"/action?go=backward")
                print("후진")
            elif key == ord('d'):
                urlopen('http://'+ ip +"/action?go=right")
                print("오른쪽")
            elif key == ord('w'):
                urlopen('http://'+ ip +"/action?go=turn_left")
                print("왼회")
            elif key == ord('w'):
                urlopen('http://'+ ip +"/action?go=turn_right")
                print("오회")                                                                
            elif key == 32:
                urlopen('http://'+ ip +"/action?go=stop")
                print("멈춰")   
            elif key == ord('1'):
                urlopen('http://'+ ip +"/action?go=speed40")
                print("40") 
            elif key == ord('2'):
                urlopen('http://'+ ip +"/action?go=speed50")
                print("50") 
            elif key == ord('3'):
                urlopen('http://'+ ip +"/action?go=speed60")
                print("60") 
            elif key == ord('4'):
                urlopen('http://'+ ip +"/action?go=speed80")
                print("80") 
            elif key == ord('5'):
                urlopen('http://'+ ip +"/action?go=speed100")
                print("100")                                                                                                     
    except:
        print("error")
        pass

urlopen('http://'+ip+"/action?go=stop")
cv2.destroyAllWindows()