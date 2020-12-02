import cv2,time,pandas
from datetime import *

frist_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])


video=cv2.VideoCapture(2)


while True:
    check ,frame=video.read()
    status=0

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   #black and white frame
    gray=cv2.GaussianBlur(gray,(21,21),0)

    if frist_frame is None:
        frist_frame=gray
        continue

    delta_frame=cv2.absdiff(frist_frame,gray)       #delta_frame
    thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]     #threshold_frame
    thresh_frame=cv2.dilate(thresh_frame,None,iterations=2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE )

    for contour in cnts:
        if cv2.contourArea(contour)<10000:
            continue
        status=1

        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    
    status_list.append(status)


    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())


    cv2.imshow("capturing",gray)
    cv2.imshow("delta_frame",delta_frame)
    cv2.imshow("threshold",thresh_frame)
    cv2.imshow("color frame",frame)

    key=cv2.waitKey(1)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break

print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("time.csv")    

video.release()
cv2.destroyAllWindows()
