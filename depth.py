
# install the required packages
# !pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# !pip install git+
# !pip install git+

#%%
#%pip install opencv-python

#%%
import cv2
import torch
import numpy as np
import os
import urllib.request

# Assuming you're working with a standard PyTorch model loading process
# Import necessary components for model transformation
from torchvision.transforms import Compose, Resize, Normalize

# Placeholder for the correct MiDaS model class import based on the specific MiDaS version
# from midas.some_module import SomeMidasModelClass

# Initialize variables
model_type = "dpt_large"  # Example model type, adjust based on the model you're working with
model_path = "model-f46da743.pt"  # Adjust to your MiDaS model weights path

# Download the MiDaS model weights if they're not already present
if not os.path.exists(model_path):
    print("Downloading the MiDaS model...")
    url = ''
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")

# Load the MiDaS model (The loading process might differ; adjust according to your MiDaS version)
midas = torch.load(model_path)
midas.eval()  # Set the model to evaluation mode

# Define a transformation pipeline for input images
transform = Compose([
    # Assuming using Resize and Normalize from torchvision.transforms
    Resize(256),  # Example resize, adjust according to model requirements
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Convert image to PyTorch tensor
    lambda image: torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float(),
])

# Example function to process an image and predict depth
def predict_depth(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image_rgb)
    
    # Predict depth
    with torch.no_grad():
        depth = midas(input_tensor)
    
    # Convert depth to numpy array and process for visualization (optional)
    depth_np = depth.squeeze().cpu().numpy()
    depth_normalized = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # Show or save the depth image
    cv2.imshow("Depth", depth_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
# predict_depth("path_to_your_image.jpg")

# %%
import cv2
import torch
import numpy as np
import os
import urllib.request

# Assuming you're working with a standard PyTorch model loading process
# Import necessary components for model transformation
from torchvision.transforms import Compose, Resize, Normalize
import sys

# %%
import numpy as np
import PIL as Image
# %%

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# %%
# Load the appropriate MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# %%
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
# %%
def real_time_depth_estimation():
    cap = cv2.VideoCapture(0)  # Open the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no frame is captured

        # Preprocess the frame for MiDaS
        input_batch = transform(frame).to(device)

        # Predict the depth
        with torch.no_grad():
            prediction = midas(input_batch)

        # Convert the depth prediction to a numpy array and process for visualization
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        prediction = cv2.normalize(
            prediction, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F
        )

        


        # Display the frame and the depth prediction
        cv2.imshow("Camera", frame)
        cv2.imshow("MiDaS", prediction)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close the windows
    cap.release()
    cv2.destroyAllWindows()

# %%
real_time_depth_estimation()

# %%
