import cv2
import numpy as np
import requests
import os
from ultralytics import YOLO

# NOTE: Before running this script, you must install the ultralytics library:
# pip install ultralytics

# Path to your t-shirt image
tshirt_path = 'blackfront.png'

# Load the t-shirt image
tshirt_orig = cv2.imread(tshirt_path, cv2.IMREAD_UNCHANGED)

if tshirt_orig is None:
    print(f"Error: Could not load the t-shirt image at '{tshirt_path}'.")
    print("Please make sure the file is in the same folder and the name is spelled correctly.")
    exit()

# Check for a transparency layer
if tshirt_orig.shape[2] < 4:
    print("Your t-shirt image is not transparent. Using an opaque version.")
    tshirt_orig = cv2.cvtColor(tshirt_orig, cv2.COLOR_BGR2BGRA)

# Load a pre-trained YOLOv8n model
# YOLOv8n is a small and fast model, ideal for real-time applications.
print("Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera. Please check if your camera is working.")
    exit()

print("Camera is open! Look at the screen and press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Perform object detection with YOLO
    # The 'stream=True' argument enables a generator for efficiency
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get the class ID and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # The COCO dataset has 'person' as class 0
            # We are only interested in 'person' detections with high confidence
            if model.names[class_id] == 'person' and confidence > 0.5:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Calculate the width and height of the detected person
                person_w = x2 - x1
                person_h = y2 - y1

                # Resize the t-shirt based on the person's dimensions
                # NOTE: The current model cannot automatically detect different body types.
                # To adjust the size for a different person, change this value.
                # A smaller value (e.g., 0.8) makes it smaller, and a larger value (e.g., 1.0) makes it bigger.
                tshirt_width = int(person_w * 0.9)
                tshirt_height = int(tshirt_orig.shape[0] * (tshirt_width / tshirt_orig.shape[1]))

                tshirt_resized = cv2.resize(tshirt_orig, (tshirt_width, tshirt_height), interpolation=cv2.INTER_AREA)

                # Check if resized t-shirt is valid
                if tshirt_resized.shape[0] == 0 or tshirt_resized.shape[1] == 0:
                    continue

                # Get the alpha mask and color channels
                alpha_mask = tshirt_resized[..., 3] / 255.0
                tshirt_color = tshirt_resized[..., :3]

                # Calculate the position to overlay the t-shirt
                tshirt_x = int(x1 + (person_w - tshirt_width) / 2)
                # Adjust this value to lower the t-shirt
                tshirt_y = int(y1 + person_h * 0.25)

                # Overlay the t-shirt on the frame
                overlay_width = tshirt_color.shape[1]
                overlay_height = tshirt_color.shape[0]

                # Ensure the overlay fits within the frame boundaries
                if tshirt_x >= 0 and tshirt_y >= 0 and tshirt_x + overlay_width <= w and tshirt_y + overlay_height <= h:
                    frame[tshirt_y:tshirt_y + overlay_height, tshirt_x:tshirt_x + overlay_width] = (
                        frame[tshirt_y:tshirt_y + overlay_height, tshirt_x:tshirt_x + overlay_width] * (1 - alpha_mask[:, :, np.newaxis]) +
                        tshirt_color * alpha_mask[:, :, np.newaxis]
                    )

    cv2.imshow('Virtual Dressing Room (YOLO)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
