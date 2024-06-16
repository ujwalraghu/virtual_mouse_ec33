import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

pyautogui.FAILSAFE = False

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

screen_width, screen_height = pyautogui.size()

def get_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0]
    return None

def hand_to_mouse(image, hand_landmarks):
    for landmark in hand_landmarks.landmark:
        if landmark.HasField('x') and landmark.HasField('y'):
            x = int(landmark.x * screen_width)
            y = int(landmark.y * screen_height)
            pyautogui.moveTo(x, y)
            break

def is_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    thumb_cmc = hand_landmarks.landmark[1]
    if (thumb_tip.y < thumb_ip.y < thumb_mcp.y < thumb_cmc.y):
        return True
    return False

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    hand_landmarks = get_hand_landmarks(frame)
    if hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        hand_to_mouse(frame, hand_landmarks)
        if is_thumb_up(hand_landmarks):
            pyautogui.click()
            print("Mouse Clicked!")
    cv2.imshow('Hand Gesture Mouse Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()