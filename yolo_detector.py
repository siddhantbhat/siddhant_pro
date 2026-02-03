from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects(frame):
    results = model(frame, conf=0.20, verbose=False)

    detected = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected.append(label)

    # Remove duplicates but KEEP multiple objects
    return list(dict.fromkeys(detected))

