import torch
from ultralytics import YOLO

# Turn to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading pretrained model YOLO v8 and turn it to GPU
model = YOLO("yolov8s-pose.pt").to(device)

# Draw conclusions on the image
results = model("C:/Users/Legion/Pictures/Saved Pictures/test.jpg")

# Show result for each
for result in results:
    result.show()
