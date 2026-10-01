import cv2
import time
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.5
CLASS_OUTPUT = {"cube": "c", "sphere": "s", "pyramid": "t"}
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

print("Loading model onto Raspberry Pi...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# ---------------------------------------------------------
# 2. CAMERA INITIALIZATION
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting real-time vision... Press 'q' to stop.")
prev_time = time.time()

# ---------------------------------------------------------
# 3. REAL-TIME DETECTION LOOP
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            # Print c for cube, s for sphere, or t for pyramid
            symbol = CLASS_OUTPUT.get(str(class_name).strip().lower())
            if symbol:
                print(symbol, flush=True)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Raspberry Pi Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()