import numpy as np
import cv2
import mediapipe as mp

nofing = [
    [6, 7, 8], #1
    [6, 7, 8, 10, 11, 12], #2
    [2, 3, 4, 6, 7, 8, 10, 11, 12], #3
    [6, 7, 8, 10, 11, 12, 14, 15, 16], #6
    [6, 7, 8, 10, 11, 12, 18, 19, 20], #7
    [6, 7, 8, 14, 15, 16, 18, 19, 20], #8
    [10, 11, 12, 14, 15, 16, 18, 19, 20], #9
    [6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]] #4

def get_points(landmark, shape, lis = None):
    points = []
    if lis==None:
        for mark in landmark:
            points.append([mark.x * shape[1], mark.y * shape[0]])
    else:
        ii = 0
        for mark in landmark:
            if ii not in nofing[lis]:
                points.append([mark.x * shape[1], mark.y * shape[0]])
            ii+=1
    
    return np.array(points, dtype=np.int32)
    


def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2)**2 + (y1 - y2) **2) **.5

def whatnum(gest):
    numget = False
    results = handsDetector.process(flippedRGB)
    while not numget:
        if results.multi_hand_landmarks is not None:
             cv2.drawContours(flippedRGB, [get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)], 0, (255, 0, 0), 2)
             (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
             ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
             print(2 * r / ws)
             if 2 * r / ws > 1.3:
                 cv2.circle(flippedRGB,(int(x), int(y)), int(r), (0, 0, 255), 2)
                 # кулак разжат
                 prev_fist = False
             else:
                 cv2.circle(flippedRGB,(int(x), int(y)), int(r), (0, 255, 0), 2)
                 if not prev_fist:
                     # произошло сжимание
                     print(0
                     # Сейчас кулак зажат
                     prev_fist = True
        # Рисуем наш результат в каждом кадре, даже если рука не детектировалась
        cv2.putText(flippedRGB, str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)

 
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
