import numpy as np
import cv2
import mediapipe as mp
from tkinter import Tk

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


scr = cv2.VideoCapture(0)
handsDetector = mp.solutions.hands.Hands()
count = 0
prev_fist = False
prev_gest = -2
outp = ""
while(scr.isOpened()):
    ret, frame = scr.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        numget = False
        cv2.drawContours(flippedRGB, [get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)], 0, (255, 0, 0), 2)
        (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
        ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
        #if 2 * r / ws > 1.3:
            #cv2.circle(flippedRGB,(int(x), int(y)), int(r), (q0, 0, 255), 2)
                # кулак разжат           
        if 2 * r / ws <= 1.3:
            cv2.circle(flippedRGB,(int(x), int(y)), int(r), (0, 255, 0), 2)
            if not prev_fist:
                     # произошло сжимание
                if prev_gest!=-1:
                    print(prev_gest, 0)
                    prev_gest = -1
                    outp+=str(0)
                prev_fist = True
                numget = True
        else: 
            ii = 0
            while not numget and ii<8:
                cv2.drawContours(flippedRGB, [get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)], 0, (255, 0, 0), 2)
                (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape, ii))
                ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
                if 2 * r / ws > 1.3:
                    #cv2.circle(flippedRGB,(int(x), int(y)), int(r), (0, 0, 255), ii)
                    prev_fist = False
                                
                else:
                    if not prev_fist:
                        cv2.circle(flippedRGB,(int(x), int(y)), int(r), (0, 255, 0), 2)
                        if prev_gest==ii:
                            break
                        else:
                            if ii<3:
                                print(ii+1)
                                outp+=str(ii+1)
                                    
                            elif ii>2 and ii<7:
                                print(ii+3)
                                outp+=str(ii+3)
                            else:
                                print(4)
                                outp+=str(4)
                            prev_gest = ii
                            prev_fist = True
                            numget = True
                ii+=1
            
                
    cv2.putText(flippedRGB, outp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)


    
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

cv2.destroyAllWindows()
r = Tk()
r.withdraw()
r.clipboard_clear()
r.clipboard_append(outp)
r.update() # now it stays on the clipboard after the window is closed
r.destroy()
