#%%
import cv2
import mediapipe as mp
import numpy as np
import csv

def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    
    return np.linalg.norm(a - b)
# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Prepare for data storage
angles_data = []
data_file = 'posture_angles.csv'

# Capture video from the webcam.
cap = cv2.VideoCapture(0)

print("Press 'c' to capture a photo and calculate angles. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and find poses.
    results = pose.process(frame_rgb)
    
    # Draw pose landmarks on the frame.
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                              connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    # Display the frame.
    cv2.imshow('MediaPipe Pose', frame)

    if cv2.waitKey(1) & 0xFF == ord('c') and results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        right_eye_outer = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y]
        right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]

        left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
        left_eye_outer = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y]
        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]

        mouth_left = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
        mouth_right = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]

        mid_mouth = [(mouth_left[0] + mouth_right[0]) / 2, (mouth_left[1] + mouth_right[1]) / 2]

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]

        diff_mouth_shoulder = calculate_distance(mid_shoulder, mid_mouth)

        diff_leye_reye = calculate_distance(left_eye, right_eye)

        diff_leare_reare = calculate_distance(left_ear, right_ear)

        # Calculate the angles
        eye_angle_left = calculate_angle(right_ear, right_eye_outer, right_eye)
        eye_angle_right = calculate_angle(left_eye, left_eye_outer, left_ear)

        # Append the features to the list
        angles_data.append([eye_angle_left, eye_angle_right, diff_mouth_shoulder, diff_leye_reye, diff_leare_reare])

        print(f"Captured angles: Angle1: {eye_angle_left}, Angle2: {eye_angle_right}")
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        # Save data to CSV file when quitting
        with open(data_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Angle1", "Angle2", "DiffMouthShoulder", "DiffLEyeREye", "DiffLEarREar"])
            writer.writerows(angles_data)
        print(f"Data saved to {data_file}")
        break

# Release the webcam and destroy all OpenCV windows.
cap.release()
cv2.destroyAllWindows()

#%%
import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
import csv
import threading

#%%
def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    
    return np.linalg.norm(a - b)
# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

class PostureApp:
    def __init__(self, root):
        self.root = root
        self.setup_gui()
        self.posture_thread = threading.Thread(target=self.posture_detection, daemon=True)
        self.running = False
        self.data_file_good = 'good_posture.csv'
        self.data_file_bad = 'bad_posture.csv'
        self.current_posture = []


    def setup_gui(self):
        self.root.title("Posture Classification")
        self.root.geometry("300x100")
        tk.Button(self.root, text="Good Posture", command=lambda: self.save_posture('good')).pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        tk.Button(self.root, text="Bad Posture", command=lambda: self.save_posture('bad')).pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    def start_posture_detection(self):
        self.running = True
        self.posture_thread.start()

    def stop_posture_detection(self):
        self.running = False
        self.posture_thread.join()

    def save_posture(self, posture_type):
        if posture_type == 'good':
            data_file = self.data_file_good
        else:
            data_file = self.data_file_bad

        # Save the current posture data to the respective file
        with open(data_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.current_posture)

    def posture_detection(self):
        # Initialize MediaPipe Pose.
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        mp_drawing = mp.solutions.drawing_utils

        # Capture video from the webcam.
        cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the image and find poses.
            results = pose.process(frame_rgb)

            # Draw pose landmarks on the frame.
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            # Display the frame.
            cv2.imshow('MediaPipe Pose', frame)

            if  results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                right_eye_outer = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y]
                right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]

                left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
                left_eye_outer = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y]
                left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]

                mouth_left = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
                mouth_right = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]

                mid_mouth = [(mouth_left[0] + mouth_right[0]) / 2, (mouth_left[1] + mouth_right[1]) / 2]

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]

                diff_mouth_shoulder = calculate_distance(mid_shoulder, mid_mouth)

                diff_leye_reye = calculate_distance(left_eye, right_eye)

                diff_leare_reare = calculate_distance(left_ear, right_ear)

                # Calculate the angles
                eye_angle_left = calculate_angle(right_ear, right_eye_outer, right_eye)

                eye_angle_right = calculate_angle(left_eye, left_eye_outer, left_ear)

                # Append the features to the list
                self.current_posture = [eye_angle_left, eye_angle_right, diff_mouth_shoulder, diff_leye_reye, diff_leare_reare]

                print(f"Captured angles: Angle1: {eye_angle_left}, Angle2: {eye_angle_right}")

        # Release the webcam and destroy all OpenCV windows.
        cap.release()
        cv2.destroyAllWindows()

#%%                                                               

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        app.stop_posture_detection()
        root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureApp(root)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app.start_posture_detection()
    root.mainloop()

# %%
