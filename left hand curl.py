import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture('test1.mp4')

start = 1 #used to show the start location
counter = 0
last_angle = None
last_stage = None
current_state = "down"
threshold = 30
peak_angle = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            # [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            # hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            # shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            # knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            angle = calculate_angle(shoulder, elbow, wrist)
            if peak_angle is None:
                peak_angle = angle
                last_angle = angle

            if start:
                print(f"{current_state} - Count: {counter}, Angle: {angle:.2f}, Peak: {peak_angle:.2f}")
                start=0
            if (current_state == "down" and angle > peak_angle) or (current_state == "up" and angle < peak_angle):
                peak_angle = angle

            # if angle > 155:
            #     stage = "down"

            # if angle < 120 and stage == 'down':
            #     stage = "up"

            #     counter += 1
            #     print("Sit-up count: ", counter)
            # if last_stage != stage:
            #     print(stage)
            #     last_stage=stage
            #     print("shoulder: ",shoulder)
            #     print("elbow: ",elbow)
            #     print("wrist: ",wrist)

            if current_state == "down" and angle < peak_angle - threshold:
                current_state = "up"
                counter += 1
                if 1:
                    print(f"{current_state} - Count: {counter}, Angle: {angle:.2f}, Peak: {peak_angle:.2f}")
                peak_angle = angle
            elif current_state == "up" and angle > peak_angle + threshold:
                current_state = "down"
                if 1:
                    print(f"{current_state} - Count: {counter}, Angle: {angle:.2f}, Peak: {peak_angle:.2f}")

            # angle_change = abs(angle - peak_angle)
            # if angle_change >= threshold:
            #     if angle > peak_angle and current_state == "down":
            #         current_state = "up"
            #         counter += 1 
        except:
            pass

        cv2.putText(image, "Counter: " + str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Sit-Up Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()