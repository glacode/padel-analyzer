from ultralytics import YOLO
import cv2
from tqdm import tqdm

# Load pre-trained YOLOv8 model
model = YOLO("yolov8x.pt")  # or yolov8n.pt for faster inference

# Open input video
video_path = 'input.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output video writer
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Define the classes you want to track (person and sports ball)
target_classes = ["person", "sports ball"]

# Get class indices from model names
class_indices = [i for i, name in model.names.items() if name in target_classes]

# Process video with tracking
for frame_idx in tqdm(range(total_frames), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking (persist=True maintains ID between frames)
    results = model.track(
        frame, 
        persist=True, 
        classes=class_indices, 
        conf=0.3, 
        verbose=False
    )
    
    # Get the first result (since we're processing one frame at a time)
    result = results[0]
    
    # Check if tracking is available
    if result.boxes.id is not None:
        # Draw detections with tracking IDs
        for box, cls, track_id in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.id):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            color = (0, 255, 0) if label == "person" else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display label and track ID
            text = f"{label} {int(track_id)}"
            cv2.putText(frame, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        # Fallback to regular detection if no tracking
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            color = (0, 255, 0) if label == "person" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write frame to output
    out.write(frame)

cap.release()
out.release()
print("Video processing complete!")