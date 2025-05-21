from ultralytics import YOLO
import cv2
from tqdm import tqdm # Import tqdm

# Load pre-trained YOLOv8 model (consider training on a tennis dataset if needed)
model = YOLO("yolov8x.pt")  # You can try yolov8n.pt for faster inference

# Open input video
video_path = 'input.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get total number of frames

# Output video writer
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Loop over video frames with tqdm for progress indication
# Wrap the loop with tqdm(total=total_frames, desc="Processing video")
for frame_idx in tqdm(range(total_frames), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model.predict(frame, conf=0.3, verbose=False)[0]

    # Draw detections
    for box in results.boxes:
        cls = int(box.cls)
        label = model.names[cls]
        if label in ["person", "sports ball"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if label == "person" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write frame to output
    out.write(frame)

cap.release()
out.release()
print("Video processing complete!")