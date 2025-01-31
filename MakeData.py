import cv2
import mediapipe as mp
import json
import os
import time

# Mediapipe setup
mphandmodul = mp.solutions.hands
hands = mphandmodul.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
mpdraw = mp.solutions.drawing_utils
captures = 0
while(1):
    print("1: OpenHand 2: CloseHand 3: ThumbsUP 4: Fcuk_off")
    user_label = int(input("Enter the Gesture you want to record: "))
    if(user_label <= 4 and user_label >= 1):
        print("Place your hand in the correct gesture in front of the camera.")
        break
    else:
        continue
if(user_label == 1):
    label = "open"
elif(user_label == 2):
    label = "close"
elif(user_label == 3):
    label = "thumbs_up"
elif(user_label == 4):
    label = "fcuk_off"
# Video capture
vid = cv2.VideoCapture(0)
data_file = r"data2.json"
gesture_data = []
prev_time = 0
interval = 0.03
temp = 0
try:
    with open(data_file, "r") as f:
        gesture_data = json.load(f)
        print(f"Loaded {len(gesture_data)} gestures from '{data_file}'.")
except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
    print(f"Error loading data: {e}. Starting fresh.")
    gesture_data = []


print("Start Moving your Hand to Record the Gesture automatically")
print("Press 'q' to quit and save the data to a JSON file.")
landmarks = []
while True:
    success, frame = vid.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for handLm in result.multi_hand_landmarks:
            landmarks = []
            for lm in handLm.landmark:
                landmarks.append([lm.x, lm.y, lm.z])  # Flattened x, y, z
            cv2.putText(frame, f"DATA_Recorded : {captures}/1200", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mpdraw.draw_landmarks(frame, handLm, mphandmodul.HAND_CONNECTIONS)

    current_time = time.time()
    cv2.imshow("Hand Gesture Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF
    if current_time - prev_time >= interval:  # Save data
        prev_time = current_time
        if landmarks != []:
            gesture_data.append({"gesture": label,"landmarks": landmarks})
        # print(f"Saved gesture: {label}")
        captures += 1
    elif key == ord('q'):  # Quit
        break
    elif captures >= 600:
        break

vid.release()
cv2.destroyAllWindows()

# Save data to JSON file
with open(r"data2.json", "w") as f:
    json.dump(gesture_data, f, indent=4)

# print("Data saved to 'gesture_data.json'")    