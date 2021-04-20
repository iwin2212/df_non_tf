import cv2
import imutils
import numpy as np
from imutils.video import FPS, FileVideoStream
import time
cap = FileVideoStream("rtsp://admin:ECSIAQ@192.168.1.48:554").start()
fps = FPS().start()
while cap.more():
    t0 = time.time()
    frame = cap.read()
    if frame is None:
        continue
    frame_resized = imutils.resize(frame, width=360)
    img = np.float32(frame_resized)
    fpstext = "FPS: " + str(1/(time.time() - t0))[:2]
    cv2.putText(frame, fpstext,
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 180, 255), 2)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()
cv2.destroyAllWindows()
cap.stop()
fps.stop()
