import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

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

left_start = 1 
left_counter = 0
left_last_angle = None
left_last_stage = None
left_current_state = "down"
left_threshold = 40
left_peak_angle = None
left_status_pos = (50, 100)
left_status_text = ""
left_status_color = (255, 255, 0)
left_angles = []

# right arm
right_start = 1 
right_counter = 0
right_last_angle = None
right_last_stage = None
right_current_state = "down"
right_threshold = 40
right_peak_angle = None
right_status_pos = (50, 140)
right_status_text = ""
right_status_color = (255, 255, 0)
right_angles = []

vis_threshold=0.5


with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # # draw picture
            # plt.figure(figsize=(12, 8))
            
            # # left
            # if len(left_angles) > 0:
            #     plt.subplot(2, 1, 1)
            #     plt.plot(range(len(left_angles)), left_angles, label='Left Arm Angle', color='blue')
            #     plt.title('Left Arm Angle Changes')
            #     plt.xlabel('Frames')
            #     plt.ylabel('Angle (degrees)')
            #     plt.grid(True)
            #     plt.legend()
            
            # # right
            # if len(right_angles) > 0:
            #     plt.subplot(2, 1, 2)
            #     plt.plot(range(len(right_angles)), right_angles, label='Right Arm Angle', color='red')
            #     plt.title('Right Arm Angle Changes')
            #     plt.xlabel('Frames')
            #     plt.ylabel('Angle (degrees)')
            #     plt.grid(True)
            #     plt.legend()
            
            # plt.tight_layout()
            # plt.show()
            break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        image.flags.writeable = False

        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        visible_arms = []
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Check visibility
            left_vis = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility
            right_vis = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility
            
            # Create a variable to track which arms are visible

            if left_vis > vis_threshold:
                visible_arms.append("left")
            if right_vis > vis_threshold:
                visible_arms.append("right")
            
            # Process left arm
            if "left" in visible_arms:
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                left_angles.append(left_angle)
                
                if left_peak_angle is None:
                    left_peak_angle = left_angle

                if left_start:
                    left_status_text = f"Left {left_current_state} - Count: {left_counter}, Angle: {left_angle:.2f}"
                    left_start = 0
                    
                if (left_current_state == "down" and left_angle > left_peak_angle) or (left_current_state == "up" and left_angle < left_peak_angle):
                    left_peak_angle = left_angle

                if left_current_state == "down" and left_angle < left_peak_angle - left_threshold:
                    left_current_state = "up"
                    left_counter += 1
                    left_status_text = f"Left {left_current_state} - Count: {left_counter}, Angle: {left_angle:.2f}"
                    left_status_color = (0, 255, 0)
                    left_peak_angle = left_angle
                elif left_current_state == "up" and left_angle > left_peak_angle + left_threshold:
                    left_current_state = "down"
                    left_status_text = f"Left {left_current_state} - Count: {left_counter}, Angle: {left_angle:.2f}"
                    left_status_color = (0, 0, 255)
            else:
                # If left arm not visible, add None to data to maintain data length consistency
                left_angles.append(None)
            
            # Process right arm
            if "right" in visible_arms:
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                right_angles.append(right_angle)
                
                if right_peak_angle is None:
                    right_peak_angle = right_angle

                if right_start:
                    right_status_text = f"Right {right_current_state} - Count: {right_counter}, Angle: {right_angle:.2f}"
                    right_start = 0
                    
                if (right_current_state == "down" and right_angle > right_peak_angle) or (right_current_state == "up" and right_angle < right_peak_angle):
                    right_peak_angle = right_angle

                if right_current_state == "down" and right_angle < right_peak_angle - right_threshold:
                    right_current_state = "up"
                    right_counter += 1
                    right_status_text = f"Right {right_current_state} - Count: {right_counter}, Angle: {right_angle:.2f}"
                    right_status_color = (0, 255, 0)
                    right_peak_angle = right_angle
                elif right_current_state == "up" and right_angle > right_peak_angle + right_threshold:
                    right_current_state = "down"
                    right_status_text = f"Right {right_current_state} - Count: {right_counter}, Angle: {right_angle:.2f}"
                    right_status_color = (0, 0, 255)
            else:
                # If right arm not visible, add None to data to maintain data length consistency
                right_angles.append(None)
                
        except Exception as e:
            # Add None to both lists to maintain data length consistency
            left_angles.append(None)
            right_angles.append(None)
            pass

        # Display detected arm status
        arm_status = "None"
        if "left" in visible_arms and "right" in visible_arms:
            arm_status = "Both Arms"
        elif "left" in visible_arms:
            arm_status = "Left Arm"
        elif "right" in visible_arms:
            arm_status = "Right Arm"
        
        cv2.putText(image, f"Detected: {arm_status}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display left and right arm counts
        cv2.putText(image, f"Left Count: {left_counter}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Display left arm status (if visible)
        if "left" in visible_arms:
            cv2.putText(image, left_status_text, left_status_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_status_color, 2, cv2.LINE_AA)
        
        # Display right arm count
        cv2.putText(image, f"Right Count: {right_counter}", (50, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Display right arm status (if visible)
        if "right" in visible_arms:
            cv2.putText(image, right_status_text, right_status_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_status_color, 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Dual Arm Exercise Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Clean None values stored in angles lists for plotting
cleaned_left_angles = [a for a in left_angles if a is not None]
cleaned_right_angles = [a for a in right_angles if a is not None]

# If program didn't end normally (e.g. by pressing q), still display charts
if len(cleaned_left_angles) > 0 or len(cleaned_right_angles) > 0:
    plt.figure(figsize=(12, 8))
    
    if len(cleaned_left_angles) > 0:
        plt.subplot(2, 1, 1)
        plt.plot(range(len(cleaned_left_angles)), cleaned_left_angles, label='Left Arm Angle', color='blue')
        plt.title('Left Arm Angle Changes')
        plt.xlabel('Frames')
        plt.ylabel('Angle (degrees)')
        plt.grid(True)
        plt.legend()
    
    if len(cleaned_right_angles) > 0:
        plt.subplot(2, 1, 2)
        plt.plot(range(len(cleaned_right_angles)), cleaned_right_angles, label='Right Arm Angle', color='red')
        plt.title('Right Arm Angle Changes')
        plt.xlabel('Frames')
        plt.ylabel('Angle (degrees)')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
