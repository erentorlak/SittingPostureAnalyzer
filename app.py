#%%
#%pip install mediapipe 

#%%
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam.
cap = cv2.VideoCapture(0)

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

    # Press 'q' to exit the loop.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows.
cap.release()
cv2.destroyAllWindows()


#%%
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(p1, p2):
    """Calculate the angle between two points."""
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi

# Capture video from the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB and process the image.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Draw pose landmarks.
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Example: Shoulder and Hip landmarks
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        # Calculate the angle.
        angle = calculate_angle(left_shoulder, right_hip)
        
        # Check if the user is slouching.
        if angle < 160:
            cv2.putText(frame, 'Please straighten your back', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, f'Angle: {int(angle)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame.
    cv2.imshow('MediaPipe Pose', frame)

    # Exit loop.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up.
cap.release()
cv2.destroyAllWindows()

# %%
import cv2
import mediapipe as mp
import numpy as np
import csv

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

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

    # Display the frame.
    cv2.imshow('MediaPipe Pose', frame)

    if cv2.waitKey(5) & 0xFF == ord('c'):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Points for first angle calculation
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            right_eye_outer = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y]
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
            
            # Points for second angle calculation
            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
            left_eye_outer = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            
            # Calculate angles
            angle1 = calculate_angle(right_ear, right_eye_outer, right_eye)
            angle2 = calculate_angle(left_eye, left_eye_outer, left_ear)
            
            angles_data.append([angle1, angle2])



            print(f"Captured angles: Angle1: {angle1}, Angle2: {angle2}")
        else:
            print("No pose detected.")

    elif cv2.waitKey(5) & 0xFF == ord('q'):
        # Save data to CSV file when quitting
        with open(data_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Angle1", "Angle2"])  # Header
            writer.writerows(angles_data)
        print(f"Data saved to {data_file}")
        break

# Release the webcam and destroy all OpenCV windows.
cap.release()
cv2.destroyAllWindows()

# %%
import cv2
import mediapipe as mp
import numpy as np
import csv

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

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

    if cv2.waitKey(2) & 0xFF == ord('c') and results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Define the points for angle calculation
        right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        right_eye_outer = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y]
        right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
        
        left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
        left_eye_outer = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y]
        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        
        # Calculate the angles
        angle1 = calculate_angle(right_ear, right_eye_outer, right_eye)
        angle2 = calculate_angle(left_eye, left_eye_outer, left_ear)
        
        #angle = (angle1 + angle2) / 2

        angles_data.append([angle1, angle2])
        print(f"Captured angles: Angle1: {angle1}, Angle2: {angle2}")

    elif cv2.waitKey(2) & 0xFF == ord('q'):
        # Save data to CSV file when quitting
        with open(data_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Angle1", "Angle2"])  # Header
            writer.writerows(angles_data)
        print(f"Data saved to {data_file}")
        break

# Release the webcam and destroy all OpenCV windows.
cap.release()
cv2.destroyAllWindows()

# %%
import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained Random Forest model
model = joblib.load('posture_model.pkl')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    
    return np.linalg.norm(a - b)




# Function to preprocess frame and extract features
def extract_features(landmarks):

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


    # Calculate the angles
    angle1 = calculate_angle(right_ear, right_eye_outer, right_eye)
    angle2 = calculate_angle(left_eye, left_eye_outer, left_ear)

    # Return the features in the same format as your training data
    return np.array([[angle1, angle2]])

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue
    
    # Process the frame with MediaPipe Pose
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    # Check if any poses are detected
    if results.pose_landmarks:
        # Extract features from the current frame
        features = extract_features(results.pose_landmarks.landmark)
        
        # Use the trained model to predict the posture class
        prediction = model.predict(features)
        
        # Display the prediction result
        posture_label = 'Good Posture' if prediction[0] == 1 else 'Bad Posture'
        cv2.putText(frame, posture_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow('MediaPipe Pose with Posture Classification', frame)
    
    # Break the loop on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# %%
