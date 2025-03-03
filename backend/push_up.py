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

def push_up():
    # Start webcam
    cap = cv2.VideoCapture(0)

    counter, position = 0, None  # Push-up count & position tracking
    last_feedback = ""  

    cv2.namedWindow("Push-Up Tracker", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Push-Up Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        feedback = "Maintain proper form!"  

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get keypoints
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

            # Calculate angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            hip_angle = calculate_angle(shoulder, hip, knee)
            knee_touching = abs(knee[1] - ankle[1]) < 0.05  

            # Push-up tracking logic
            if elbow_angle > 160:  
                position = "UP"
            elif elbow_angle < 90 and position == "UP":  
                position = "DOWN"
                counter += 1
                last_feedback = speak_once(f"Push-up {counter}", last_feedback)  

            # Feedback conditions
            if elbow_angle > 160:
                feedback = "Lower more!"
            elif elbow_angle < 90:
                feedback = "Good! Now push up."
            elif hip_angle > 160:
                feedback = "Keep your body straight!"
            elif hip_angle < 130:
                feedback = "Avoid raising hips!"
            elif knee_touching:
                feedback = "Don't touch your knee!"
            else:
                feedback = "Maintain a steady pace."

            # Speak only when feedback changes
            last_feedback = speak_once(feedback, last_feedback)

            # Display keypoints and feedback
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f'Push-ups: {counter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, feedback, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Maximize window and display
        cv2.imshow("Push-Up Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
