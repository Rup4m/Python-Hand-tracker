import cv2
import mediapipe as mp
import os
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Finger tip landmarks (based on MediaPipe's hand landmarks)
tip_ids = [4, 8, 12, 16, 20]

shutdown_triggered = False
countdown_start = None

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    fingers = []

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, _ = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Thumb
            if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other four fingers
            for id in range(1, 5):
                if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)

            cv2.putText(img, f'Fingers: {total_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if total_fingers == 3 and not shutdown_triggered:
                countdown_start = time.time()
                shutdown_triggered = True

            # If 3 fingers held up for 3 seconds
            if shutdown_triggered and time.time() - countdown_start >= 3:
                print("Shutting down...")
                os.system("shutdown /s /t 1")  # For Windows
                # os.system("sudo shutdown now")  # For Linux
                break

            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
