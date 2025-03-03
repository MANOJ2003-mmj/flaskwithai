import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import multiprocessing


# Initialize Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Initialize Text-to-Speech
engine = pyttsx3.init()


def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Speak function to avoid event loop issues
def speak_once(text, last_feedback):
    if text != last_feedback:
        process = multiprocessing.Process(target=speak_text, args=(text,))
        process.start()
    return text


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

def plank():
        
    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Maximize screen width
    cap.set(4, 720)   # Maximize screen height

    last_feedback = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        feedback = "Hold steady!"

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get key points
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]

            # Calculate angles
            back_angle = calculate_angle(shoulder, hip, knee)
            leg_angle = calculate_angle(hip, knee, ankle)
            arm_angle = calculate_angle(shoulder, elbow, wrist=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y])

            # Give feedback based on angles
            if back_angle < 165:
                feedback = f"Raise your hips by {int(165 - back_angle)} degrees!"
            elif back_angle > 175:
                feedback = f"Lower your hips by {int(back_angle - 175)} degrees!"
            elif arm_angle < 75:
                feedback = f"Straighten your arms by {int(75 - arm_angle)} degrees!"
            elif leg_angle < 170:
                feedback = f"Straighten your legs by {int(170 - leg_angle)} degrees!"
            else:
                feedback = "Perfect plank position!"

            last_feedback = speak_once(feedback, last_feedback)

            # Draw landmarks and feedback
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, feedback, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Plank Posture Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
