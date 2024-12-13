import cv2
import time

import HandTrackingModule as htm

# def main():

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHand(img,draw = True)
    lmlist = detector.findPosition(img, draw = True)

    if len(lmlist) != 0:

        # the specific point should be provide by user
        print(lmlist[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


# if __name__ == "__main__":
#     main()
