#%%
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk

class PostureDetector:
    """
    Class for detecting and analyzing sitting posture using MediaPipe and OpenCV.

    Attributes:
        master (Tk): The Tkinter master window.
        mp_pose (mp.solutions.pose): MediaPipe Pose object for pose estimation.
        pose (mp.solutions.pose.Pose): MediaPipe Pose object for pose estimation.
        mp_drawing (mp.solutions.drawing_utils): MediaPipe DrawingUtils object for drawing landmarks.
        cap (cv2.VideoCapture): OpenCV VideoCapture object for capturing video frames.
        good_posture_file (str): File path for the CSV file to store data for good posture.
        bad_posture_file (str): File path for the CSV file to store data for bad posture.
        column_names (list): List of column names for the CSV file.
        video_label (Label): Tkinter Label widget for displaying the video frame.
        current_results (mp.solutions.pose.Pose): Current pose estimation results.
    """

    def __init__(self, master):
        """
        Initializes the PostureDetector object.

        Args:
            master (Tk): The Tkinter master window.
        """
        self.master = master
        self.master.title("Posture Classification")
        self.setup_gui()

        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

        # CSV file setup
        self.good_posture_file = 'csv_files/goodPosture.csv'
        self.bad_posture_file = 'csv_files/badPosture.csv'

    def setup_gui(self):
        """
        Sets up the GUI elements for displaying the video frame and buttons.
        """
        self.video_label = Label(self.master)
        self.video_label.pack()

        Button(self.master, text="Good Posture", command=lambda: self.save_posture('good')).pack(side='left')
        Button(self.master, text="Bad Posture", command=lambda: self.save_posture('bad')).pack(side='right')
        Button(self.master, text="Quit", command=self.cleanup_and_close).pack(side='bottom')
    
    def cleanup_and_close(self):
        """
        Releases the webcam and closes the application window properly.
        """
        if self.cap.isOpened():
            self.cap.release()  # Release the webcam
        cv2.destroyAllWindows()  # Close all OpenCV windows
        self.master.destroy()  # Close the Tkinter GUI window
    
    def save_posture(self, posture_type):
        if hasattr(self, 'current_results') and self.current_results.pose_landmarks:
            features_dict = self.extract_angles(self.current_results.pose_landmarks.landmark)
            
            # Dynamically get column names from the features_dict keys
            column_names = list(features_dict.keys())
            features_values = list(features_dict.values())

            data_file = self.good_posture_file if posture_type == 'good' else self.bad_posture_file

            # Check if file exists to decide whether to write headers
            write_header = not os.path.exists(data_file)
            with open(data_file, 'a', newline='') as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow(column_names)
                writer.writerow(features_values)

            print(f"Saved {posture_type} posture features to {data_file} and features: {features_values}")

    def update_video_frame(self, frame):
        """
        Updates the video frame displayed in the GUI.

        Args:
            frame (numpy.ndarray): The video frame as a NumPy array.
        """
        # Convert the OpenCV image to a PIL image and display it in Tkinter 
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        img = img.resize((640, 480), Image.LANCZOS)  # Image.LANCZOS is a high-quality downsampling filter for resizing
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk  # Keep a reference! Otherwise, it may be garbage collected  
        self.video_label.configure(image=imgtk)

    def process_frame(self):
        """
        Processes a single video frame for pose estimation and updates the GUI.

        Returns:
            bool: True if the frame was processed successfully, False otherwise.
        """
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
        """
        Runs the posture detection and GUI update loop.
        """
        # Use after to schedule the next frame processing, ensuring it runs on the main thread
        if self.process_frame():
            self.master.after(30, self.run)  # Schedule the next run

    def extract_angles(self, landmarks):
       """
       Extracts the angles between specific landmarks for posture analysis.
    
       Args:
           landmarks (List[mp.solutions.pose.PoseLandmark]): List of pose landmarks.
    
       Returns:
           List[float]: List of extracted angles and distances.
       """
       lm = lambda p: [landmarks[p.value].x, landmarks[p.value].y]
    
       # Extracting specific landmarks
       right_ear = lm(mp.solutions.pose.PoseLandmark.RIGHT_EAR)
       right_eye_outer = lm(mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER)
       right_eye = lm(mp.solutions.pose.PoseLandmark.RIGHT_EYE)
       left_eye = lm(mp.solutions.pose.PoseLandmark.LEFT_EYE)
       left_eye_outer = lm(mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER)
       left_ear = lm(mp.solutions.pose.PoseLandmark.LEFT_EAR)
       mouth_left = lm(mp.solutions.pose.PoseLandmark.MOUTH_LEFT)
       mouth_right = lm(mp.solutions.pose.PoseLandmark.MOUTH_RIGHT)
       left_shoulder = lm(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
       right_shoulder = lm(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)
    
       # Calculating angles and distances
       right_eye_angle = self.calculate_angle(right_ear, right_eye_outer, right_eye)
       left_eye_angle = self.calculate_angle(left_eye, left_eye_outer, left_ear)
       distance_mouth_shoulder = self.calculate_distance(
           np.mean([mouth_left, mouth_right], axis=0),
           np.mean([left_shoulder, right_shoulder], axis=0)
       )
       distance_leye_reye = self.calculate_distance(left_eye, right_eye)
       distance_leare_reare = self.calculate_distance(left_ear, right_ear)
    
       # Returning features as a dictionary to dynamically generate column names later
       return {
           "Right Eye Angle": right_eye_angle,
           "Left Eye Angle": left_eye_angle,
           "Distance Mouth to Shoulder": distance_mouth_shoulder,
           "Distance Left Eye to Right Eye": distance_leye_reye,
           "Distance Left Ear to Right Ear": distance_leare_reare
        
            # TODO: you can add more features here for generalization 

            #,"Ratio":( distance_leye_reye + distance_leare_reare/2) / distance_mouth_shoulder
       }

    @staticmethod
    def calculate_distance(a, b):
        """
        Calculates the Euclidean distance between two points.
        """
        return np.linalg.norm(np.array(a) - np.array(b))
    
    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculates the angle between three points using the arctan2 function.
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

if __name__ == "__main__":
    root = Tk()
    root.geometry("800x600")
    pd = PostureDetector(root)
    root.after(0, pd.run)  # Start the video processing loop
    root.protocol("WM_DELETE_WINDOW", pd.cleanup_and_close)  # Ensure cleanup is called when window is closed
    root.mainloop()
    
# %%
