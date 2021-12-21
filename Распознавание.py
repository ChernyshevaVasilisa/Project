import numpy as np
import cv2
import mediapipe as mp
from tkinter import Tk

nofing = [
    [6, 7, 8], #1
    [6, 7, 8, 10, 11, 12], #2
    [2, 3, 4, 6, 7, 8, 10, 11, 12], #3
    [2, 3, 4, 6, 7, 8, 18,19, 20], #5
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

print("Добро пожаловать! Эта программа будет считывать ваши жесты и выводить те цифры, которые вы показываете, а после закрытия программы полученнная строка цифр скопируется в ваш буфер обмена. Используются жесты жестового языка, так что вы можете показывать одной рукой любые цифры от нуля до девяти включительно (чтобы показать пять, покажите 'козу'). Для лучшего распознавания,советую после каждой цифры показывать раскрытую ладонь")
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
        #cv2.drawContours(flippedRGB, [get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)], 0, (255, 0, 0), 2)
        (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
        ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
        #if 2 * r / ws > 1.3:
            #cv2.circle(flippedRGB,(int(x), int(y)), int(r), (q0, 0, 255), 2)          
        if 2 * r / ws <= 1.3:
            if not prev_fist:
                if prev_gest!=-1:
                    prev_gest = -1
                    outp+=str(0)
                prev_fist = True
                numget = True
        else: 
            ii = 0
            while not numget and ii<9:
                #cv2.drawContours(flippedRGB, [get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)], 0, (255, 0, 0), 2)
                (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape, ii))
                ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
                if 2 * r / ws > 1.3:
                    prev_fist = False
                                
                else:
                    if not prev_fist:
                        numget = True
                        if prev_gest!= ii:
                            if ii<3:
                                outp+=str(ii+1)
                                    
                            elif ii>2 and ii<8:
                                outp+=str(ii+2)

                            else:
                                outp+="4"
                            prev_gest = ii
                            prev_fist = True
                            
                
                
                ii+=1
                
    if len(outp)<16:
        cv2.putText(flippedRGB, outp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
            
    elif len(outp)>15 and len(outp)<31:
        outp1 = ""
        outp2 = ""
        for i in range(len(outp)-15):
            outp2+=outp[i+15]
        for i in range(15):
            outp1+=outp[i]
        cv2.putText(flippedRGB, outp1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        cv2.putText(flippedRGB, outp2, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        
    elif len(outp)>30 and len(outp)<46:
        outp1 = ""
        outp2 = ""
        outp3 = ""
        for i in range(15):
            outp1+=outp[i]
        for i in range(15):
            outp2+=outp[i+15]
        for i in range(len(outp)-30):
            outp3+=outp[i+30]
        cv2.putText(flippedRGB, outp1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        cv2.putText(flippedRGB, outp2, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        cv2.putText(flippedRGB, outp3, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        
    elif len(outp)>45 and len(outp)<61:
        outp1 = ""
        outp2 = ""
        outp3 = ""
        outp4 = ""
        for i in range(15):
            outp1+=outp[i]
        for i in range(15):
            outp2+=outp[i+15]
        for i in range(15):
            outp3+=outp[i+30]
        for i in range(len(outp)-45):
            outp4+=outp[i+45]
        cv2.putText(flippedRGB, outp1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        cv2.putText(flippedRGB, outp2, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        cv2.putText(flippedRGB, outp3, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        cv2.putText(flippedRGB, outp4, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        
    else:
        outp1 = ""
        outp2 = ""
        outp3 = ""
        outp4 = ""
        outp5 = ""
        for i in range(15):
            outp1+=outp[i]
        for i in range(15):
            outp2+=outp[i+15]
        for i in range(15):
            outp3+=outp[i+30]
        for i in range(15):
            outp4+=outp[i+45]
        for i in range(len(outp)-60):
            outp5+=outp[i+60]
        cv2.putText(flippedRGB, outp1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        cv2.putText(flippedRGB, outp2, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        cv2.putText(flippedRGB, outp3, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        cv2.putText(flippedRGB, outp4, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        cv2.putText(flippedRGB, outp5, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
    
    
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

cv2.destroyAllWindows()
r = Tk()
r.withdraw()
r.clipboard_clear()
r.clipboard_append(outp)
r.update() # now it stays on the clipboard after the window is closed
r.destroy()
