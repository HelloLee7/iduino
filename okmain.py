from urllib.request import urlopen
import cv2
import numpy as np

ip = '192.168.137.107'
stream = urlopen('http://' + ip + ':81/stream')
buffer = b''

while True:
    buffer += stream.read(4096)
    head = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')

    try:
        if head > -1 and end > -1:
            jpg = buffer[head:end+2]
            buffer = buffer[end+2:]
            # dtype으로 수정
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            cv2.imshow("AI car stream", img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    except:
        print("error")
        pass

cv2.destroyAllWindows()
