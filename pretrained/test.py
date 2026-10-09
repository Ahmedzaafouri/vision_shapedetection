"""Zero-shot shape detection with a PRETRAINED YOLO-World model (no training).

Classes: Cube, Sphere, Triangle (= pyramid / triangle / cone), Square, Circle.

  python test.py                    # camera 1 (usually the external webcam)
  python test.py 0 0.25             # camera 0, base conf 0.25

Keys:
  q = quit
  s = save frame to ./captures
  d = DEBUG mode: shows EVERY raw detection (very low conf, no merging) with the
      exact prompt that fired. Use it on the pyramid to see what the model thinks.
"""
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLOWorld

CAMERA =0
BASE_CONF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
CUBE_MIN_CONF = 0.25
ROI_X = 0.15   # left edge
ROI_Y = 0.50   # top edge
ROI_W = 0.70   # width
ROI_H = 0.40   # height

# Text prompt -> final class. Several words can point to the same class: the model
# understands some words much better than others ("ball" is usually easier than "sphere").
PROMPT_TO_CLASS = {
    "cube": "Cube", "box": "Cube",
    "sphere": "Sphere", "ball": "Sphere",
    "pyramid": "Triangle", "triangle": "Triangle", "cone": "Triangle",
    "square": "Square",
    "circle": "Circle",
}
PROMPTS = list(PROMPT_TO_CLASS)

# Per-class confidence threshold. Cube gets a higher floor to suppress weak false
# positives; Triangle gets a lower one because abstract pyramids score lower.
CONF_BY_CLASS = {
    "Cube": max(BASE_CONF, CUBE_MIN_CONF),
    "Sphere": BASE_CONF, "Square": BASE_CONF, "Circle": BASE_CONF,
    "Triangle": BASE_CONF / 2,
}
CLASS_TO_LETTER = {"Cube": "c", "Sphere": "s", "Triangle": "t", "Square": "c", "Circle": "s"}
COLORS = {
    "Cube": (255, 0, 0), "Sphere": (255, 255, 0), "Triangle": (0, 255, 0),
    "Square": (0, 165, 255), "Circle": (255, 0, 255),
}
MERGE_IOU = 0.5   # two boxes overlapping more than this = same object, keep the best one
DEBUG_CONF = 0.02

model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(PROMPTS)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit(f"Cannot open camera {CAMERA}")
Path("captures").mkdir(exist_ok=True)

debug = False
prev = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    raw = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    roi_x1 = int(frame_width * ROI_X)
    roi_y1 = int(frame_height * ROI_Y)
    roi_x2 = int(frame_width * (ROI_X + ROI_W))
    roi_y2 = int(frame_height * (ROI_Y + ROI_H))
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    r = model.predict(roi, conf=DEBUG_CONF if debug else min(CONF_BY_CLASS.values()),
                      verbose=False)[0]

    # All raw detections: (confidence, prompt, final class, box)
    dets = []
    for box in r.boxes:
        prompt = PROMPTS[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        dets.append((float(box.conf[0]), prompt, PROMPT_TO_CLASS[prompt],
                     (x1 + roi_x1, y1 + roi_y1, x2 + roi_x1, y2 + roi_y1)))
    dets.sort(key=lambda d: d[0], reverse=True)

    roi_center_x = (roi_x1 + roi_x2) / 2
    roi_center_y = (roi_y1 + roi_y2) / 2

    def center_priority(detection):
        confidence, _, _, (x1, y1, x2, y2) = detection
        distance = ((x1 + x2) / 2 - roi_center_x) ** 2 + ((y1 + y2) / 2 - roi_center_y) ** 2
        return distance, -confidence

    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)
    if debug:
        # Show the strongest raw detections with the exact prompt that fired
        for i, (c, prompt, cls, (x1, y1, x2, y2)) in enumerate(dets[:8]):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(frame, f"{prompt} {c:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        eligible = [d for d in dets if d[0] >= CONF_BY_CLASS[d[2]]]
        if eligible:
            selected = min(eligible, key=center_priority)
            print(CLASS_TO_LETTER[selected[2]], flush=True)
    else:
        # Keep detections above their class threshold, then merge duplicates
        kept = []
        for c, prompt, cls, b in dets:
            if c < CONF_BY_CLASS[cls]:
                continue
            if any(iou(b, k[3]) > MERGE_IOU for k in kept):
                continue
            kept.append((c, prompt, cls, b))
        if kept:
            selected = min(kept, key=center_priority)
            c, prompt, cls, (x1, y1, x2, y2) = selected
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[cls], 2)
            cv2.putText(frame, f"{cls} {c:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls], 2)
            print(CLASS_TO_LETTER[selected[2]], flush=True)

    now = time.time()
    cv2.putText(frame, f"FPS: {1 / max(now - prev, 1e-6):.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if debug:
        cv2.putText(frame, "DEBUG (d to leave)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    prev = now

    cv2.imshow("YOLO-World test", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("d"):
        debug = not debug
    if key == ord("s"):
        path = f"captures/{int(time.time())}.jpg"
        cv2.imwrite(path, raw)

cap.release()
cv2.destroyAllWindows()