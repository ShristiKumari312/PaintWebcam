
from pickle import FRAME, NONE
from tkinter import Frame
import numpy as np
import cv2
from collections import deque

bluelower = np.array([100, 60, 60])
blueupper = np.array([140, 255, 255])

kernel = np.ones((5, 5),np.uint8)

bPoints = [deque(maxlen=512)]
rPoints = [deque(maxlen=512)]
yPoints = [deque(maxlen=512)]
gPoints = [deque(maxlen=512)]

bindex = 0
rindex = 0
yindex = 0
gindex = 0

color = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(0, 255, 255)]
colorindex = 0

paintwindow = np.zeros((471, 630, 3))+255
paintwindow = cv2.rectangle(paintwindow, (40,1),(140,65),(0,0,0),2)
paintwindow = cv2.rectangle(paintwindow, (160,1),(255,65),color[0],-1)
paintwindow = cv2.rectangle(paintwindow, (275,1),(370,65),color[1],-1)
paintwindow = cv2.rectangle(paintwindow, (390,1),(485,65),color[2],-1)
paintwindow = cv2.rectangle(paintwindow, (505,1),(600,65),color[3],-1)

cv2.putText(paintwindow, "clear all",(49,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintwindow, "BLUE",(185,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(paintwindow, "GREEN",(298,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(paintwindow, "RED",(420,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(paintwindow, "YELLOW",(520,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),2,cv2.LINE_AA)

cv2.namedWindow('Paint',cv2.WINDOW_AUTOSIZE)
camera=cv2.VideoCapture(0)
while True :
    (grab, Frame)=camera.read()
    Frame=cv2.flip(Frame,1)
    hsv=cv2.cvtColor(Frame,cv2.COLOR_BGR2HSV)

    Frame = cv2.rectangle(Frame, (40,1),(140,65),(0,0,0),2)
    Frame = cv2.rectangle(Frame, (160,1),(255,65),color[0],-1)
    Frame = cv2.rectangle(Frame, (275,1),(370,65),color[1],-1 )
    Frame = cv2.rectangle(Frame, (390,1),(485,65),color[2],-1)
    Frame = cv2.rectangle(Frame, (505,1),(600,65),color[3],-1)
    cv2.putText(Frame, "clear all",(49,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(Frame, "BLUE",(185,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(Frame, "GREEN",(298,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(Frame, "RED",(420,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(Frame, "YELLOW",(520,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),2,cv2.LINE_AA)

    if not grab:
        break

    bluemask=cv2.inRange(hsv, bluelower, blueupper)
    bluemask=cv2.erode(bluemask, kernel, iterations=2)
    bluemask=cv2.morphologyEx(bluemask,cv2.MORPH_OPEN,kernel)
    bluemask=cv2.dilate(bluemask,kernel,iterations=1)
    (cnts, _)=cv2.findContours(bluemask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center= None
    if len(cnts) > 0:
        cnt=sorted(cnts,key=cv2.contourArea,reverse=True)[0]
        ((x, y),radius)=cv2.minEnclosingCircle(cnt)
        cv2.circle(Frame,(int(x),int(y)),int(radius),(0,255,255),2)
        n=cv2.moments(cnt)
        center= (int(n['m10'] / n['m00']), int(n['m01'] / n['m00']))
        if center[1]<=65:
            if 40<=center[0]<=140:
                bPoints=[deque(maxlen=512)]
                rPoints=[deque(maxlen=512)]
                yPoints=[deque(maxlen=512)]
                gPoints=[deque(maxlen=512)]
                bindex = 0
                gindex = 0
                rindex = 0
                yindex = 0
                paintwindow[67:,:,:]=255
            elif 160<=center[0]<=255:
                colorindex=0
            elif 275<=center[0]<=370:
                colorindex=1
            elif 390<=center[0]<=485:
                colorindex=2
            elif 505<=center[0]<=600:
                colorindex=3
        else :
            if colorindex == 0:
                bPoints[bindex].appendleft(center)
            elif colorindex == 1:
                gPoints[gindex].appendleft(center)
            elif colorindex == 2:
                rPoints[rindex].appendleft(center)
            elif colorindex == 3:
                yPoints[yindex].appendleft(center)
    else :
        bPoints.append(deque(maxlen=512))
        bindex+=1
        rPoints.append(deque(maxlen=512))
        rindex+=1
        yPoints.append(deque(maxlen=512))
        yindex+=1
        gPoints.append(deque(maxlen=512))
        gindex+=1
    points=[bPoints,gPoints, rPoints, yPoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1,len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(Frame,points[i][j][k-1], points[i][j][k], color[i],2)
                cv2.line(paintwindow,points[i][j][k-1], points[i][j][k], color[i],2)
                

                
    cv2.imshow('Paint',paintwindow)
    cv2.imshow('tracking', Frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

camera.release()
cv2.destroyAllWindows() 