import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Initialize Pose and Text-to-Speech
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
engine = pyttsx3.init()

# Speak function to prevent repeated speech
def speak_once(text, last_feedback):
    if text != last_feedback:
        engine.say(text)
        engine.runAndWait()
    return text

# Function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Maximize screen width
cap.set(4, 720)   # Maximize screen height

count = 0
position = "Down"
last_feedback = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    feedback = "Raise your hands up!"

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get key points
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

        # Calculate angles
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Shoulder Press Logic
        if left_arm_angle > 160 and right_arm_angle > 160:  # Raised
            if position == "Down":
                count += 1
            position = "Up"
            feedback = "Lower your hands down!"
        elif left_arm_angle < 90 and right_arm_angle < 90:  # Lowered
            position = "Down"
            feedback = "Push your hands up!"

        last_feedback = speak_once(feedback, last_feedback)

        # Draw landmarks
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display count and feedback
        cv2.putText(frame, f"Shoulder Press Count: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, feedback, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Shoulder Press Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
