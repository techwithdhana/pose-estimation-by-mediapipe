import cv2


import poseModule as pm
 
cap = cv2.VideoCapture('sources/2.jpg')
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) !=0:        
        cv2.circle(img, (lmList[0][1], lmList[0][2]), 5, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)
    cv2.waitKey(0)

