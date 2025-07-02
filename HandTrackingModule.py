import cv2
import mediapipe as mp

class handDetector():

    def __init__(self, mode =False, maxHands=2, model_complexity=1, detectionConfidence=0.5, trackingConfidence=0.5):
        # static_image_mode = False,
        # max_num_hands = 2,
        # model_complexity = 1,
        # min_detection_confidence = 0.5,
        # min_tracking_confidence = 0.5)

        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = model_complexity
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence,
                                        self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHand(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handsLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handsLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, HandNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[HandNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList
# the dump code
