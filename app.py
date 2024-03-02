#%%
# %pip install mediapipe --user
# %%
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd

class PostureClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils

    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculates the angle between three points using the arctan2 function.
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    @staticmethod
    def calculate_distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def extract_features(self, landmarks):
        lm = lambda p: [landmarks[p.value].x, landmarks[p.value].y]
        right_ear, right_eye_outer, right_eye = lm(mp.solutions.pose.PoseLandmark.RIGHT_EAR), lm(mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER), lm(mp.solutions.pose.PoseLandmark.RIGHT_EYE)
        left_eye, left_eye_outer, left_ear = lm(mp.solutions.pose.PoseLandmark.LEFT_EYE), lm(mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER), lm(mp.solutions.pose.PoseLandmark.LEFT_EAR)
        mouth_left, mouth_right = lm(mp.solutions.pose.PoseLandmark.MOUTH_LEFT), lm(mp.solutions.pose.PoseLandmark.MOUTH_RIGHT)
        left_shoulder, right_shoulder = lm(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER), lm(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)

        mid_mouth = np.mean([mouth_left, mouth_right], axis=0)
        mid_shoulder = np.mean([left_shoulder, right_shoulder], axis=0)

        features = [
            self.calculate_angle(right_ear, right_eye_outer, right_eye),    # Right eye angle
            self.calculate_angle(left_eye, left_eye_outer, left_ear),       # Left eye angle    
            self.calculate_distance(mid_shoulder, mid_mouth),               # Distance mouth to shoulder
            self.calculate_distance(left_eye, right_eye),                   # Distance left eye to right eye
            self.calculate_distance(left_ear, right_ear)                    # Distance left ear to right ear

            # TODO: you can add more features here for generalization 

            #,"Ratio":( distance_leye_reye + distance_leare_reare/2) / distance_mouth_shoulder
        ]
        return features
    

    def classify_posture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            # Extract features and create a DataFrame with the correct column names
            features = self.extract_features(results.pose_landmarks.landmark)
            feature_df = pd.DataFrame([features], columns=['Eye Angle Right', 'Eye Angle Left', 'Distance Mouth to Shoulder', 'Distance Left Eye to Right Eye', 'Distance Left Ear to Right Ear'])

            # Use the DataFrame to make predictions
            prediction = self.model.predict(feature_df)
            label = 'Good Posture' if prediction[0] == 1 else 'Bad Posture'
            color = (0, 255, 0) if label == 'Good Posture' else (0, 0, 255)
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        return frame

def main():
    classifier = PostureClassifier('posture_model.pkl')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = classifier.classify_posture(frame)
        cv2.imshow('MediaPipe Pose with Posture Classification', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# %%
