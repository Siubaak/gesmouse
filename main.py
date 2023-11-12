import cv2
import pyautogui
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 240)
cap.set(4, 320)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
screenWidth, screenHeight = pyautogui.size()
clicking = 0

def mouseMove(x, y):
    curX, curY = pyautogui.position()
    moveX, moveY = max(min(screenWidth * (3 - (x * 5)), screenWidth - 1), 0), min(screenHeight * (1 - y * 3), screenHeight - 1)
    if abs(moveX - curX) > 5 or abs(moveY - curY) > 5:
        pyautogui.moveTo(moveX, moveY)


def mouseClick(baseX, baseY, x, y):
    global clicking
    delta = (baseX - x) ** 2 + (baseY - y) ** 2
    if delta < 0.002 and clicking == 0:
        pyautogui.mouseDown()
        clicking = 1
    elif delta > 0.0021 and clicking == 1:
        pyautogui.mouseUp()
        clicking = 0


while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                moveBase = handLms.landmark[0]
                clickBase = handLms.landmark[4]
                leftClkBase = handLms.landmark[8]
                mouseMove(moveBase.x, moveBase.y)
                mouseClick(clickBase.x, clickBase.y, leftClkBase.x, leftClkBase.y)

        cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
