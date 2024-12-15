import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("D:\FCI\FCI_Fourth_Year\CVision\FCI-Computer-Vision-Tasks\Task_3\yolov8n.pt")  # Ensure this is the correct model for your task

# Path to the input image
image_path = "D:\FCI\FCI_Fourth_Year\CVision\FCI-Computer-Vision-Tasks\Task_3\lion6.jpg"
output_path = "D:\FCI\FCI_Fourth_Year\CVision\FCI-Computer-Vision-Tasks\Task_3\lion_detected6.jpg"

# Verify the image exists
import os
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Failed to load image: {image_path}")

# Perform object detection
results = model(image_path, conf=0.25)  # Lower the confidence threshold

# Debugging: Print results
print("Detection Results:")
print(results)

# Check if any detections were made
if not results[0].boxes:
    print("No objects detected!")
else:
    # Iterate over detections and draw bounding boxes
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        confidence = detection.conf[0]
        class_id = int(detection.cls[0])

        # Print detection details
        print(f"Detected class: {model.names[class_id]}, Confidence: {confidence:.2f}")

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{model.names[class_id]}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save and display the output
    cv2.imwrite(output_path, image)
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
