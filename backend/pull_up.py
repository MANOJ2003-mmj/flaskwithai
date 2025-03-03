import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import multiprocessing
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Initialize Text-to-Speech Engine
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

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))
def pullup():
    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Set width
    cap.set(4, 720)   # Set height

    counter, position = 0, None  # Pull-up count & position tracking
    last_feedback = ""  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        feedback = "Maintain proper form!"  # Default feedback

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get keypoints
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

            # Calculate elbow angle
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            # Pull-up tracking logic
            if elbow_angle > 160:  # Down position
                position = "DOWN"
            elif elbow_angle < 90 and position == "DOWN":  # Up position
                position = "UP"
                counter += 1
                last_feedback = speak_once(f"Pull-up {counter}", last_feedback)

            # Feedback conditions
            if elbow_angle > 160:
                feedback = "Pull yourself up!"
            elif elbow_angle < 90:
                feedback = "Good! Lower down slowly."
            else:
                feedback = "Maintain control!"

            last_feedback = speak_once(feedback, last_feedback)

            # Display keypoints and feedback
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f'Pull-ups: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, feedback, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Pull-Up Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
