# ---- Monkey patch for Windows ----
import pathlib
if pathlib.Path().__class__.__name__ != "PosixPath":
    pathlib.PosixPath = pathlib.WindowsPath





import torch
import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device


import os
os.environ["TORCH_HOME"] = str(Path.home())  # Ensures paths work correctly on Windows


import sys
sys.path.append(str(Path(r"D:\prof-project\yolov5")))  # Adjust the path to your YOLOv5 folder
from yolov5.models.common import DetectMultiBackend

"""
# Load YOLOv5 trained model
device = select_device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

"""


# Load YOLOv5 trained model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model = DetectMultiBackend(str(Path(r"D:\prof-project\best_weights.pt")), device=device)  # Ensure 'best.pt' is your trained model
model.eval()  # Set to evaluation mode

# Define the class index for screws (update this based on your dataset)
# SCREW_CLASS_ID = 1  # Change this if necessary

# Define class names for screws (update based on your dataset)
CLASS_NAMES = ["Screw Type A", "Screw Type B", "Screw Type C", "Screw Type D"]

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream depth and color images
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for frames and get both depth and color frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue  # Skip iteration if no frame is available

        # Convert images to NumPy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert to tensor for YOLOv5
        img = torch.from_numpy(color_image).to(device)
        img = img.permute(2, 0, 1).float() / 255.0  # Convert to (C, H, W) and normalize
        img = img.unsqueeze(0)  # Add batch dimension

        # Perform YOLO inference
        pred = model(img, augment=False)
        pred = non_max_suppression(pred, 0.5, 0.45)  # Apply NMS

        # Draw bounding boxes for detected screws (specific to 4 screw types)
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    if conf < 0.9:  # Skip predictions with confidence less than 90%
                        continue
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    class_name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else f"Class {int(cls)}"
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Apply color map to depth image for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack color and depth images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Display the images
        cv2.imshow("Intel RealSense D435i - Screw Detection", images)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()





# the detections is based on the following four classes:
# cross , hexa (predominant) (has cross and slotted also), slotted , nut


# the dataset has now changed to the following classes:
# cross (has some slotted and hexa also) (predominant) , slotted , nut



