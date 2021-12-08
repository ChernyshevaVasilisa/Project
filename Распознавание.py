import numpy as np
import cv2
import mediapipe as mp

def whatnum(gest):
    
    


 
scr = cv2.VideoCapture(0)

while(scr.isOpened()):
    ret, frame = scr.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    flippedRGB = cv# Распознаем
    results = handsDetector.process(flippedRGB)
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)
    
cv2.destroyAllWindows()
