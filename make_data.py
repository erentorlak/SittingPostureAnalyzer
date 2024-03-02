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

        # initialize columns for the CSV file
        with open(self.data_file_good, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Angle1", "Angle2", "DiffMouthShoulder", "DiffLEyeREye", "DiffLEarREar"])


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
import cv2
import numpy as np
import mediapipe as mp
import threading
import csv
import tkinter as tk
from tkinter import messagebox

def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
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

        # initialize columns for the CSV file
        with open(self.data_file_good, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Angle1", "Angle2", "DiffMouthShoulder", "DiffLEyeREye", "DiffLEarREar"])

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
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        mp_drawing = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            cv2.imshow('MediaPipe Pose', frame)

            if results.pose_landmarks:
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

                eye_angle_left = calculate_angle(right_ear, right_eye_outer, right_eye)
                eye_angle_right = calculate_angle(left_eye, left_eye_outer, left_ear)

                self.current_posture = [eye_angle_left, eye_angle_right, diff_mouth_shoulder, diff_leye_reye, diff_leare_reare]

                print(f"Captured angles: Angle1: {eye_angle_left}, Angle2: {eye_angle_right}")
                

        cap.release()
        cv2.destroyAllWindows()

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
import cv2
import mediapipe as mp
import numpy as np
import csv

# Utility Functions
def calculate_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)

def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) between three points with 'b' as the vertex."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def extract_angles(landmarks):
    """Extract and calculate required angles and distances from landmarks."""
    # Simplified landmark extraction for readability
    lm = lambda p: [landmarks[p.value].x, landmarks[p.value].y]
    right_ear, right_eye_outer, right_eye = lm(mp_pose.PoseLandmark.RIGHT_EAR), lm(mp_pose.PoseLandmark.RIGHT_EYE_OUTER), lm(mp_pose.PoseLandmark.RIGHT_EYE)
    left_eye, left_eye_outer, left_ear = lm(mp_pose.PoseLandmark.LEFT_EYE), lm(mp_pose.PoseLandmark.LEFT_EYE_OUTER), lm(mp_pose.PoseLandmark.LEFT_EAR)
    mouth_left, mouth_right = lm(mp_pose.PoseLandmark.MOUTH_LEFT), lm(mp_pose.PoseLandmark.MOUTH_RIGHT)
    left_shoulder, right_shoulder = lm(mp_pose.PoseLandmark.LEFT_SHOULDER), lm(mp_pose.PoseLandmark.RIGHT_SHOULDER)

    # Distance and angle calculations
    mid_mouth = np.mean([mouth_left, mouth_right], axis=0)
    mid_shoulder = np.mean([left_shoulder, right_shoulder], axis=0)
    return [
        calculate_angle(right_ear, right_eye_outer, right_eye),
        calculate_angle(left_eye, left_eye_outer, left_ear),
        calculate_distance(mid_shoulder, mid_mouth),
        calculate_distance(left_eye, right_eye),
        calculate_distance(left_ear, right_ear)
    ]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Data Storage Preparation
angles_data = []
data_file = 'posture_angles.csv'

# Video Capture Setup
cap = cv2.VideoCapture(0)
print("Press 'c' to capture a photo and calculate angles. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue  # Skip empty frames

    # Image Processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Drawing Landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    cv2.imshow('MediaPipe Pose', frame)

    # Capture and Calculate Angles on "b" Press (for "bad" posture) or "g" Press (for "good" posture) 
    if cv2.waitKey(1) & ( 0xFF == ord('b') or 0xFF == ord('g') or 0xFF == ord('q') ):
        if results.pose_landmarks:

            


# Save Data to CSV
with open(data_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Angle1", "Angle2", "DiffMouthShoulder", "DiffLEyeREye", "DiffLEarREar"])
    writer.writerows(angles_data)
print(f"Data saved to {data_file}")

# Cleanup
cap.release()
cv2.destroyAllWindows()

# %%
import cv2
import mediapipe as mp
import numpy as np
import csv

class PostureDetector:
    def __init__(self, data_file='posture_angles.csv'):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.data_file = data_file
        self.angles_data = []

    @staticmethod
    def calculate_distance(a, b):
        """Calculate the Euclidean distance between two points."""
        return np.linalg.norm(np.array(a) - np.array(b))

    @staticmethod
    def calculate_angle(a, b, c):
        """Calculate the angle (in degrees) between three points with 'b' as the vertex."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def extract_features(self, landmarks):
        """Extract and calculate required angles and distances from landmarks."""
        features = {}
        # Define points
        points = {
            'right_ear': landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value],
            'right_eye_outer': landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value],
            'right_eye': landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE.value],
            'left_eye': landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value],
            'left_eye_outer': landmarks[self.mp_pose.PoseLandmark.LEFT_EYE_OUTER.value],
            'left_ear': landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value],
            'mouth_left': landmarks[self.mp_pose.PoseLandmark.MOUTH_LEFT.value],
            'mouth_right': landmarks[self.mp_pose.PoseLandmark.MOUTH_RIGHT.value],
            'left_shoulder': landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            'right_shoulder': landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        }

        # Convert to x, y coordinates
        for key, point in points.items():
            features[key] = [point.x, point.y]

        # Calculate distances and angles
        features['diff_mouth_shoulder'] = self.calculate_distance(
            np.mean([features['mouth_left'], features['mouth_right']], axis=0),
            np.mean([features['left_shoulder'], features['right_shoulder']], axis=0)
        )
        features['diff_leye_reye'] = self.calculate_distance(features['left_eye'], features['right_eye'])
        features['diff_leare_reare'] = self.calculate_distance(features['left_ear'], features['right_ear'])
        features['eye_angle_left'] = self.calculate_angle(features['right_ear'], features['right_eye_outer'], features['right_eye'])
        features['eye_angle_right'] = self.calculate_angle(features['left_eye'], features['left_eye_outer'], features['left_ear'])

        return features

    def capture_posture(self):
        print("Press 'g' for good posture, 'b' for bad posture, 'q' to quit.")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                cv2.imshow('MediaPipe Pose', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('g') or key == ord('b'):
                    features = self.extract_features(results.pose_landmarks.landmark)
                    posture_type = 'good' if key == ord('g') else 'bad'
                    self.save_data(features, posture_type)
                    print(f"Captured {posture_type} posture features.")
                elif key == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    def save_data(self, features, posture_type):
        with open(self.data_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(features.keys() + ['posture_type'])
            writer.writerow(features.values() + [posture_type])

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.capture_posture()

if __name__ == "__main__":
    detector = PostureDetector()
    detector.run()


# %%
import cv2
import mediapipe as mp
import numpy as np
import csv

class PostureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.angles_data = []
        self.good_posture_file = 'goodPosture.csv'
        self.bad_posture_file = 'badPosture.csv'

    def calculate_distance(self, a, b):
        a, b = np.array(a), np.array(b)
        return np.linalg.norm(a - b)

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def extract_features(self, landmarks):
        features = {}
        points = {
            'right_ear': landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value],
            'right_eye_outer': landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value],
            'right_eye': landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE.value],
            'left_eye': landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value],
            'left_eye_outer': landmarks[self.mp_pose.PoseLandmark.LEFT_EYE_OUTER.value],
            'left_ear': landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value],
            'mouth_left': landmarks[self.mp_pose.PoseLandmark.MOUTH_LEFT.value],
            'mouth_right': landmarks[self.mp_pose.PoseLandmark.MOUTH_RIGHT.value],
            'left_shoulder': landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            'right_shoulder': landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        }
        for key, point in points.items():
            features[key] = [point.x, point.y]
        mid_mouth = np.mean([features['mouth_left'], features['mouth_right']], axis=0)
        mid_shoulder = np.mean([features['left_shoulder'], features['right_shoulder']], axis=0)
        features['diff_mouth_shoulder'] = self.calculate_distance(mid_mouth, mid_shoulder)
        features['diff_leye_reye'] = self.calculate_distance(features['left_eye'], features['right_eye'])
        features['diff_leare_reare'] = self.calculate_distance(features['left_ear'], features['right_ear'])
        features['eye_angle_left'] = self.calculate_angle(features['right_ear'], features['right_eye_outer'], features['right_eye'])
        features['eye_angle_right'] = self.calculate_angle(features['left_eye'], features['left_eye_outer'], features['left_ear'])
        return [features['eye_angle_left'], features['eye_angle_right'], features['diff_mouth_shoulder'], features['diff_leye_reye'], features['diff_leare_reare']]

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                cv2.imshow('MediaPipe Pose', frame)
                if cv2.waitKey(1) & 0xFF == ord('g'):
                    features = self.extract_features(results.pose_landmarks.landmark)
                    self.save_data(features, self.good_posture_file)
                elif cv2.waitKey(1) & 0xFF == ord('b'):
                    features = self.extract_features(results.pose_landmarks.landmark)
                    self.save_data(features, self.bad_posture_file)
                elif cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def save_data(self, features, file_name):
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(features)
        print(f"Data saved to {file_name}")

if __name__ == "__main__":
    detector = PostureDetector()
    detector.run()

# %%
import cv2
import mediapipe as mp
import numpy as np
import csv

# Utility Functions
def calculate_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)

def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) between three points with 'b' as the vertex."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def extract_angles(landmarks):
    """Extract and calculate required angles and distances from landmarks."""
    # Simplified landmark extraction for readability
    lm = lambda p: [landmarks[p.value].x, landmarks[p.value].y]
    right_ear, right_eye_outer, right_eye = lm(mp.solutions.pose.PoseLandmark.RIGHT_EAR), lm(mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER), lm(mp.solutions.pose.PoseLandmark.RIGHT_EYE)
    left_eye, left_eye_outer, left_ear = lm(mp.solutions.pose.PoseLandmark.LEFT_EYE), lm(mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER), lm(mp.solutions.pose.PoseLandmark.LEFT_EAR)
    mouth_left, mouth_right = lm(mp.solutions.pose.PoseLandmark.MOUTH_LEFT), lm(mp.solutions.pose.PoseLandmark.MOUTH_RIGHT)
    left_shoulder, right_shoulder = lm(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER), lm(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)

    # Distance and angle calculations
    mid_mouth = np.mean([mouth_left, mouth_right], axis=0)
    mid_shoulder = np.mean([left_shoulder, right_shoulder], axis=0)
    return [
        calculate_angle(right_ear, right_eye_outer, right_eye),
        calculate_angle(left_eye, left_eye_outer, left_ear),
        calculate_distance(mid_shoulder, mid_mouth),
        calculate_distance(left_eye, right_eye),
        calculate_distance(left_ear, right_ear)
    ]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Data Storage Preparation
angles_data = []
good_posture_file = 'goodPosture.csv'
bad_posture_file = 'badPosture.csv'

# Video Capture Setup
cap = cv2.VideoCapture(0)
print("Press 'g' for good posture, 'b' for bad posture, and 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue  # Skip empty frames

    # Image Processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Drawing Landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    cv2.imshow('MediaPipe Pose', frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('b'), ord('g')]:
        if results.pose_landmarks:
            features = extract_angles(results.pose_landmarks.landmark)
            data_file = good_posture_file if key == ord('g') else bad_posture_file
            with open(data_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(features)
            print(f"Features saved to {data_file}.")
    elif key == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()


# %%
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Utility Functions
def calculate_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)

def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) between three points with 'b' as the vertex."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def extract_angles(landmarks, mp_pose):
    """Extract and calculate required angles and distances from landmarks."""
    # Simplified landmark extraction for readability
    lm = lambda p: [landmarks[p.value].x, landmarks[p.value].y]
    right_ear, right_eye_outer, right_eye = lm(mp_pose.PoseLandmark.RIGHT_EAR), lm(mp_pose.PoseLandmark.RIGHT_EYE_OUTER), lm(mp_pose.PoseLandmark.RIGHT_EYE)
    left_eye, left_eye_outer, left_ear = lm(mp_pose.PoseLandmark.LEFT_EYE), lm(mp_pose.PoseLandmark.LEFT_EYE_OUTER), lm(mp_pose.PoseLandmark.LEFT_EAR)
    mouth_left, mouth_right = lm(mp_pose.PoseLandmark.MOUTH_LEFT), lm(mp_pose.PoseLandmark.MOUTH_RIGHT)
    left_shoulder, right_shoulder = lm(mp_pose.PoseLandmark.LEFT_SHOULDER), lm(mp_pose.PoseLandmark.RIGHT_SHOULDER)

    # Distance and angle calculations
    mid_mouth = np.mean([mouth_left, mouth_right], axis=0)
    mid_shoulder = np.mean([left_shoulder, right_shoulder], axis=0)
    return [
        calculate_angle(right_ear, right_eye_outer, right_eye),
        calculate_angle(left_eye, left_eye_outer, left_ear),
        calculate_distance(mid_shoulder, mid_mouth),
        calculate_distance(left_eye, right_eye),
        calculate_distance(left_ear, right_ear)
    ]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Data Storage Preparation
good_posture_file = 'goodPosture.csv'
bad_posture_file = 'badPosture.csv'
column_names = ["Eye Angle Right", "Eye Angle Left", "Distance Mouth to Shoulder", "Distance Left Eye to Right Eye", "Distance Left Ear to Right Ear"]

# Video Capture Setup
cap = cv2.VideoCapture(0)
print("Press 'g' for good posture, 'b' for bad posture, and 'q' to quit.")

# initialize columns for the CSV file
with open(good_posture_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(column_names)
with open(bad_posture_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(column_names)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue  # Skip empty frames

    # Image Processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Drawing Landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    cv2.imshow('MediaPipe Pose', frame)

    key = cv2.waitKey(1) & 0xFF


    if key in [ord('b'), ord('g')]:
        if results.pose_landmarks:
            features = extract_angles(results.pose_landmarks.landmark, mp_pose)
            data_file = good_posture_file if key == ord('g') else bad_posture_file
            with open(data_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(features)
            print(f"Features saved to {data_file}.")
    elif key == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

# %%
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

class PostureDetector:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils

        # Data Storage Preparation
        self.good_posture_file = 'goodPosture.csv'
        self.bad_posture_file = 'badPosture.csv'
        self.column_names = ["Eye Angle Right", "Eye Angle Left", "Distance Mouth to Shoulder", "Distance Left Eye to Right Eye", "Distance Left Ear to Right Ear"]

        # Video Capture Setup
        self.cap = cv2.VideoCapture(0)
        print("Press 'g' for good posture, 'b' for bad posture, and 'q' to quit.")
        
        # Initialize CSV files with column names
        self.initialize_csv_files()

    def calculate_distance(self, a, b):
        """Calculate the Euclidean distance between two points."""
        a, b = np.array(a), np.array(b)
        return np.linalg.norm(a - b)

    def calculate_angle(self, a, b, c):
        """Calculate the angle (in degrees) between three points with 'b' as the vertex."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def extract_angles(self, landmarks):
        """Extract and calculate required angles and distances from landmarks."""
        # Simplified landmark extraction for readability
        lm = lambda p: [landmarks[p.value].x, landmarks[p.value].y]
        right_ear, right_eye_outer, right_eye = lm(self.mp_pose.PoseLandmark.RIGHT_EAR), lm(self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER), lm(self.mp_pose.PoseLandmark.RIGHT_EYE)
        left_eye, left_eye_outer, left_ear = lm(self.mp_pose.PoseLandmark.LEFT_EYE), lm(self.mp_pose.PoseLandmark.LEFT_EYE_OUTER), lm(self.mp_pose.PoseLandmark.LEFT_EAR)
        mouth_left, mouth_right = lm(self.mp_pose.PoseLandmark.MOUTH_LEFT), lm(self.mp_pose.PoseLandmark.MOUTH_RIGHT)
        left_shoulder, right_shoulder = lm(self.mp_pose.PoseLandmark.LEFT_SHOULDER), lm(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)

        # Distance and angle calculations
        mid_mouth = np.mean([mouth_left, mouth_right], axis=0)
        mid_shoulder = np.mean([left_shoulder, right_shoulder], axis=0)
        return [
            self.calculate_angle(right_ear, right_eye_outer, right_eye),
            self.calculate_angle(left_eye, left_eye_outer, left_ear),
            self.calculate_distance(mid_shoulder, mid_mouth),
            self.calculate_distance(left_eye, right_eye),
            self.calculate_distance(left_ear, right_ear)
        ]

    def initialize_csv_files(self):
        """Initialize CSV files with column names if they do not exist."""
        for file_path in [self.good_posture_file, self.bad_posture_file]:
            if not os.path.exists(file_path):
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.column_names)

    def run(self):
        """Run the posture detection and data collection process."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue  # Skip empty frames

            # Image Processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            # Drawing Landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                               landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                               connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            cv2.imshow('MediaPipe Pose', frame)

            key = cv2.waitKey(1) & 0xFF

            if key in [ord('b'), ord('g')]:
                if results.pose_landmarks:
                    features = self.extract_angles(results.pose_landmarks.landmark)
                    data_file = self.good_posture_file if key == ord('g') else self.bad_posture_file
                    with open(data_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(features)
                    print(f"Features saved to {data_file}.")
            elif key == ord('q'):
                break

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PostureDetector()
    detector.run()

# %%
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import tkinter as tk
from threading import Thread

class PostureDetector:
    def __init__(self, master):
        self.master = master
        self.setup_gui()
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

        self.good_posture_file = 'goodPosture.csv'
        self.bad_posture_file = 'badPosture.csv'
        self.column_names = ["Eye Angle Right", "Eye Angle Left", "Distance Mouth to Shoulder", "Distance Left Eye to Right Eye", "Distance Left Ear to Right Ear"]
        
        self.initialize_csv_files()

    def setup_gui(self):
        self.master.title("Posture Classification")
        tk.Button(self.master, text="Good Posture", command=lambda: self.save_posture('good')).pack(side=tk.LEFT)
        tk.Button(self.master, text="Bad Posture", command=lambda: self.save_posture('bad')).pack(side=tk.RIGHT)

    def initialize_csv_files(self):
        for file_path in [self.good_posture_file, self.bad_posture_file]:
            if not os.path.exists(file_path):
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.column_names)

    def save_posture(self, posture_type):
        if self.current_results and self.current_results.pose_landmarks:
            features = self.extract_angles(self.current_results.pose_landmarks.landmark)
            data_file = self.good_posture_file if posture_type == 'good' else self.bad_posture_file
            with open(data_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(features)
            print(f"Saved {posture_type} posture features to {data_file}")
        else:
            print("No posture detected to save.")

    def run(self):
        mp_pose = mp.solutions.pose  # Add this line to reference the module for POSE_CONNECTIONS

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue  # Skip empty frames

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_results = self.pose.process(frame_rgb)

            if self.current_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    self.current_results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,  # Use mp_pose.POSE_CONNECTIONS here
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            cv2.imshow('Posture Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()



    def extract_angles(self, landmarks):
        """Extract and calculate required angles and distances from landmarks."""
        # Simplified landmark extraction for readability
        lm = lambda p: [landmarks[p.value].x, landmarks[p.value].y]
        right_ear, right_eye_outer, right_eye = lm(self.mp_pose.PoseLandmark.RIGHT_EAR), lm(self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER), lm(self.mp_pose.PoseLandmark.RIGHT_EYE)
        left_eye, left_eye_outer, left_ear = lm(self.mp_pose.PoseLandmark.LEFT_EYE), lm(self.mp_pose.PoseLandmark.LEFT_EYE_OUTER), lm(self.mp_pose.PoseLandmark.LEFT_EAR)
        mouth_left, mouth_right = lm(self.mp_pose.PoseLandmark.MOUTH_LEFT), lm(self.mp_pose.PoseLandmark.MOUTH_RIGHT)
        left_shoulder, right_shoulder = lm(self.mp_pose.PoseLandmark.LEFT_SHOULDER), lm(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)

        # Distance and angle calculations
        mid_mouth = np.mean([mouth_left, mouth_right], axis=0)
        mid_shoulder = np.mean([left_shoulder, right_shoulder], axis=0)
        return [
            self.calculate_angle(right_ear, right_eye_outer, right_eye),
            self.calculate_angle(left_eye, left_eye_outer, left_ear),
            self.calculate_distance(mid_shoulder, mid_mouth),
            self.calculate_distance(left_eye, right_eye),
            self.calculate_distance(left_ear, right_ear)
        ]

if __name__ == "__main__":
    root = tk.Tk()
    pd = PostureDetector(root)
    
    # Run the posture detection in a separate thread to prevent GUI freeze
    thread = Thread(target=pd.run)
    thread.daemon = True
    thread.start()

    root.mainloop()

# %%
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import tkinter as tk
from threading import Thread

class PostureDetector:
    def __init__(self, master):
        self.master = master
        self.setup_gui()
        # It's crucial to keep a reference to the module for accessing POSE_CONNECTIONS
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

        self.good_posture_file = 'goodPosture.csv'
        self.bad_posture_file = 'badPosture.csv'
        self.column_names = ["Eye Angle Right", "Eye Angle Left", "Distance Mouth to Shoulder", "Distance Left Eye to Right Eye", "Distance Left Ear to Right Ear"]
        
        self.initialize_csv_files()

    @staticmethod
    def calculate_distance(a, b):
        """Calculate the Euclidean distance between two points."""
        a, b = np.array(a), np.array(b)
        return np.linalg.norm(a - b)    
    
    @staticmethod
    def calculate_angle(a, b, c):
        """Calculate the angle (in degrees) between three points with 'b' as the vertex."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def setup_gui(self):
        self.master.title("Posture Classification")
        tk.Button(self.master, text="Good Posture", command=lambda: self.save_posture('good')).pack(side=tk.LEFT)
        tk.Button(self.master, text="Bad Posture", command=lambda: self.save_posture('bad')).pack(side=tk.RIGHT)

    def initialize_csv_files(self):
        # Initialize CSV files with column names if they do not exist
        for file_path in [self.good_posture_file, self.bad_posture_file]:
            if not os.path.exists(file_path):
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.column_names)

    def save_posture(self, posture_type):
        # Save the captured posture to the appropriate CSV file
        if hasattr(self, 'current_results') and self.current_results.pose_landmarks:
            features = self.extract_angles(self.current_results.pose_landmarks.landmark)
            data_file = self.good_posture_file if posture_type == 'good' else self.bad_posture_file
            with open(data_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(features)
            print(f"Saved {posture_type} posture features to {data_file}")
        else:
            print("No posture detected to save.")

    def run(self):
        # Main loop for capturing frames and processing pose landmarks
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue  # Skip empty frames
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_results = self.pose.process(frame_rgb)
    
            if self.current_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    self.current_results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,  # Correctly use POSE_CONNECTIONS from mp_pose
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
    
            cv2.imshow('Posture Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        self.cap.release()
        cv2.destroyAllWindows()

    def extract_angles(self, landmarks):
        """Extract and calculate required angles and distances from landmarks."""
        # Simplified landmark extraction for readability
        lm = lambda p: [landmarks[p.value].x, landmarks[p.value].y]
        right_ear, right_eye_outer, right_eye = lm(self.mp_pose.PoseLandmark.RIGHT_EAR), lm(self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER), lm(self.mp_pose.PoseLandmark.RIGHT_EYE)
        left_eye, left_eye_outer, left_ear = lm(self.mp_pose.PoseLandmark.LEFT_EYE), lm(self.mp_pose.PoseLandmark.LEFT_EYE_OUTER), lm(self.mp_pose.PoseLandmark.LEFT_EAR)
        mouth_left, mouth_right = lm(self.mp_pose.PoseLandmark.MOUTH_LEFT), lm(self.mp_pose.PoseLandmark.MOUTH_RIGHT)
        left_shoulder, right_shoulder = lm(self.mp_pose.PoseLandmark.LEFT_SHOULDER), lm(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)

        # Distance and angle calculations
        mid_mouth = np.mean([mouth_left, mouth_right], axis=0)
        mid_shoulder = np.mean([left_shoulder, right_shoulder], axis=0)
        return [
            self.calculate_angle(right_ear, right_eye_outer, right_eye),
            self.calculate_angle(left_eye, left_eye_outer, left_ear),
            self.calculate_distance(mid_shoulder, mid_mouth),
            self.calculate_distance(left_eye, right_eye),
            self.calculate_distance(left_ear, right_ear)
        ]

if __name__ == "__main__":
    root = tk.Tk()
    pd = PostureDetector(root)
    
    # Run the posture detection in a separate thread to prevent GUI freeze
    thread = Thread(target=pd.run)
    thread.daemon = True
    thread.start()

    root.mainloop()

# %%
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread

class PostureDetector:
    def __init__(self, master):
        self.master = master
        self.master.title("Posture Classification")

        # Setup video frame label
        self.video_label = tk.Label(master)
        self.video_label.pack()

        # Setup GUI buttons
        self.setup_gui()

        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

        # File setup
        self.good_posture_file = 'goodPosture.csv'
        self.bad_posture_file = 'badPosture.csv'
        self.column_names = ["Eye Angle Right", "Eye Angle Left", "Distance Mouth to Shoulder", "Distance Left Eye to Right Eye", "Distance Left Ear to Right Ear"]
        
        # Initialize CSV files with column names
        self.initialize_csv_files()

    def setup_gui(self):
        # Setup GUI buttons for good and bad posture
        tk.Button(self.master, text="Good Posture", command=lambda: self.save_posture('good')).pack(side=tk.LEFT)
        tk.Button(self.master, text="Bad Posture", command=lambda: self.save_posture('bad')).pack(side=tk.RIGHT)

    def initialize_csv_files(self):
        # Initialize CSV files with column names if they do not exist
        for file_path in [self.good_posture_file, self.bad_posture_file]:
            if not os.path.exists(file_path):
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.column_names)

    def save_posture(self, posture_type):
        # Save the captured posture to the appropriate CSV file
        if hasattr(self, 'current_results') and self.current_results.pose_landmarks:
            features = self.extract_angles(self.current_results.pose_landmarks.landmark)
            data_file = self.good_posture_file if posture_type == 'good' else self.bad_posture_file
            with open(data_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(features)
            print(f"Saved {posture_type} posture features to {data_file}")
        else:
            print("No posture detected to save.")

    def update_video_frame(self, frame):
        # Convert the OpenCV image to a PIL image and display it in Tkinter
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break  # Exit if unable to read the video frame
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_results = self.pose.process(frame_rgb)
    
            if self.current_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    self.current_results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            self.update_video_frame(frame)  # Update the GUI with the new frame

        self.cap.release()

    def extract_angles(self, landmarks):
        # Simplified landmark extraction for readability
        lm = lambda p: [landmarks[p.value].x, landmarks[p.value].y]
        right_ear, right_eye_outer, right_eye = lm(self.mp_pose.PoseLandmark.RIGHT_EAR), lm(self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER), lm(self.mp_pose.PoseLandmark.RIGHT_EYE)
        left_eye, left_eye_outer, left_ear = lm(self.mp_pose.PoseLandmark.LEFT_EYE), lm(self.mp_pose.PoseLandmark.LEFT_EYE_OUTER), lm(self.mp_pose.PoseLandmark.LEFT_EAR)
        mouth_left, mouth_right = lm(self.mp_pose.PoseLandmark.MOUTH_LEFT), lm(self.mp_pose.PoseLandmark.MOUTH_RIGHT)
        left_shoulder, right_shoulder = lm(self.mp_pose.PoseLandmark.LEFT_SHOULDER), lm(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)

        # Distance and angle calculations
        mid_mouth = np.mean([mouth_left, mouth_right], axis=0)
        mid_shoulder = np.mean([left_shoulder, right_shoulder], axis=0)
        return [
            self.calculate_angle(right_ear, right_eye_outer, right_eye),
            self.calculate_angle(left_eye, left_eye_outer, left_ear),
            self.calculate_distance(mid_shoulder, mid_mouth),
            self.calculate_distance(left_eye, right_eye),
            self.calculate_distance(left_ear, right_ear)
        ]
    
#%%
    
 

# %%
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk
from threading import Thread

class PostureDetector:
    def __init__(self, master):
        self.master = master
        self.master.title("Posture Classification")
        self.setup_gui()

        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

        # CSV file setup
        self.good_posture_file = 'goodPosture.csv'
        self.bad_posture_file = 'badPosture.csv'
        self.column_names = ["Eye Angle Right", "Eye Angle Left", "Distance Mouth to Shoulder", "Distance Left Eye to Right Eye", "Distance Left Ear to Right Ear"]
        self.initialize_csv_files()

    def setup_gui(self):
        self.video_label = Label(self.master)
        self.video_label.pack()

        Button(self.master, text="Good Posture", command=lambda: self.save_posture('good')).pack(side='left')
        Button(self.master, text="Bad Posture", command=lambda: self.save_posture('bad')).pack(side='right')

    def initialize_csv_files(self):
        for file_path in [self.good_posture_file, self.bad_posture_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.column_names)

    def save_posture(self, posture_type):
        if hasattr(self, 'current_results') and self.current_results.pose_landmarks:
            features = self.extract_angles(self.current_results.pose_landmarks.landmark)
            data_file = self.good_posture_file if posture_type == 'good' else self.bad_posture_file
            with open(data_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(features)
            print(f"Saved {posture_type} posture features to {data_file}")
            # print(features)
            print(features)

        else:
            print("No posture detected to save.")

    def update_video_frame(self, frame):
        # Convert the OpenCV image to a PIL image and display it in Tkinter
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        img = img.resize((640, 480), Image.LANCZOS)  # Updated to use Image.LANCZOS
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk  # Keep a reference!
        self.video_label.configure(image=imgtk)

    def process_frame(self):
        # Separated frame processing from the run method
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()  # Release the capture if the frame couldn't be read
            return False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_results = self.pose.process(frame_rgb)
        if self.current_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                self.current_results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        self.master.after_idle(self.update_video_frame, frame)  # Schedule update on the main thread
        return True
    
    def run(self):
        # Use after to schedule the next frame processing, ensuring it runs on the main thread
        if self.process_frame():
            self.master.after(30, self.run)  # Schedule the next run

    def extract_angles(self, landmarks):
        lm = lambda p: [landmarks[p.value].x, landmarks[p.value].y]
        right_ear, right_eye_outer, right_eye = lm(self.mp_pose.PoseLandmark.RIGHT_EAR), lm(self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER), lm(self.mp_pose.PoseLandmark.RIGHT_EYE)
        left_eye, left_eye_outer, left_ear = lm(self.mp_pose.PoseLandmark.LEFT_EYE), lm(self.mp_pose.PoseLandmark.LEFT_EYE_OUTER), lm(self.mp_pose.PoseLandmark.LEFT_EAR)
        mouth_left, mouth_right = lm(self.mp_pose.PoseLandmark.MOUTH_LEFT), lm(self.mp_pose.PoseLandmark.MOUTH_RIGHT)
        left_shoulder, right_shoulder = lm(self.mp_pose.PoseLandmark.LEFT_SHOULDER), lm(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)

        mid_mouth = np.mean([mouth_left, mouth_right], axis=0)
        mid_shoulder = np.mean([left_shoulder, right_shoulder], axis=0)
        return [
            self.calculate_angle(right_ear, right_eye_outer, right_eye),
            self.calculate_angle(left_eye, left_eye_outer, left_ear),
            self.calculate_distance(mid_shoulder, mid_mouth),
            self.calculate_distance(left_eye, right_eye),
            self.calculate_distance(left_ear, right_ear)
        ]
    
    @staticmethod
    def calculate_distance(a, b):
        a, b = np.array(a), np.array(b)
        return np.linalg.norm(a - b)
    
    @staticmethod
    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle
    # Your extract_angles, calculate_distance, and calculate_angle methods remain unchanged

if __name__ == "__main__":
    root = Tk()
    root.geometry("800x600")  # Adjust the window size as needed
    pd = PostureDetector(root)
    root.after(0, pd.run)  # Start the video processing loop
    root.mainloop()

    
# %%
