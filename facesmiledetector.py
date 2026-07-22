import cv2
from random import randrange
facedata=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
smiledata=cv2.CascadeClassifier("haarcascade_smile.xml") 
camera=cv2.VideoCapture(0)
while True:
    #read single frame from the webcam 
    success, frame = camera.read()
    #converts the captured frame to grey scale(becayse face detection works better on gray scale image)    
    greyscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #it returns list of retangles with(x,y,width,height) for each detected face(face coordinates)
    facesqaure=facedata.detectMultiScale(greyscale)
    #draw retangles around each detected phase.
    for (x,y,w,h) in facesqaure:
        #use random colors and thickness of 2 for the retangle 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 2)
        #extract the frame region from the color frame 
        theface=frame[y:y+h,x:x+w]
        #converting the entract face into the gray scale to identify a smile detection
        faceimage=cv2.cvtColor(theface,cv2.COLOR_BGR2GRAY)
        #detect smile in the face region
        smile=smiledata.detectMultiScale(faceimage,scaleFactor=1.7,minNeighbors=20)
        #scale factors means image size reduction and require min neighbor rectangle for detection.

        #optionally draw retangles around the smile - testing if the code is checking the mouth
        for(x1,y1,w1,h1) in smile:
            cv2.rectangle(theface,(x1,y1), (x1+w1, y1+h1),(255,0,0), 2)

        #displace smile detection result 
        if len(smile)>0:
            cv2.putText(frame,"SMILING",(x,y+h+40),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(0,255,0),thickness=2)
        else:
            cv2.putText(frame,"NOT SMILING",(x,y+h+40),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,0,0),thickness=2)
    print(facesqaure)
        #cv2.putText(image,text,org,fontFace,fontScale,color,thickness)

    cv2.imshow("Face Detector",frame)
    #wait one millisecond for key press
    key=cv2.waitKey(1)
    if key==81 or key==113:
        camera.release()
        cv2.destroyAllWindows()
        break


