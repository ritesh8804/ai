import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from deepface import DeepFace
import time
from datetime import datetime
import os

# === CONFIG ===
REFERENCE_IMAGE_PATH = "reference.jpg"
EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

CALIBRATION_TIME = 4  # seconds
ALERT_THRESHOLD = 10  # seconds for empty chair

# YOLO model
yolo_model = YOLO("yolov8n.pt")  # pre-trained YOLOv8

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Head and Eye calibration offsets
head_calibrated = False
eye_calibrated = False
head_pitch_offset, head_yaw_offset = 0, 0
left_eye_offset, right_eye_offset = 0, 0

# Empty chair detection
last_seen_time = time.time()

# Webcam
cap = cv2.VideoCapture(0)
w, h = 640, 480


# Logging function
def log_event(event, frame=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[{timestamp}] {event}")
    if frame is not None:
        cv2.imwrite(f"{EVIDENCE_DIR}/{timestamp}_{event}.jpg", frame)


# --- STEP 1: Identity Verification ---
verified = False
while not verified:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, (w, h))
    cv2.putText(frame, "Show your face for verification", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Identity Verification", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        exit()
    elif key == ord('c'):  # press 'c' to capture
        cv2.imwrite("capture.jpg", frame)
        try:
            result = DeepFace.verify("capture.jpg", REFERENCE_IMAGE_PATH, enforce_detection=True)
            confidence = result["distance"]
            if result["verified"]:
                verified = True
                print(f"✅ Identity Verified | Distance: {confidence:.2f}")
            else:
                print(f"❌ Not Verified | Distance: {confidence:.2f}")
        except Exception as e:
            print("Error:", e)

cv2.destroyWindow("Identity Verification")

# --- STEP 2: Proctoring Detection ---
calibration_start = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (w, h))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    # YOLO Detection
    results = yolo_model(frame)
    detections = results[0].boxes.data
    person_count = 0
    cellphone_detected = False
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        label = results[0].names[int(cls)]
        if label == "person" and conf > 0.5:
            person_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        if label in ["cell phone", "mobile"] and conf > 0.5:
            cellphone_detected = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            log_event("Cell Phone Detected", frame)

    # FaceMesh detection
    mp_results = face_mesh.process(rgb)
    face_present = mp_results.multi_face_landmarks is not None

    # --- Empty Chair Detection ---
    if person_count == 1 and not face_present and time.time() - last_seen_time > ALERT_THRESHOLD:
        log_event("Empty Chair Detected", frame)
        last_seen_time = time.time()
    elif face_present:
        last_seen_time = time.time()

    # --- Calibration Phase ---
    if face_present and (not head_calibrated or not eye_calibrated):
        if calibration_start is None:
            calibration_start = time.time()
        elapsed = time.time() - calibration_start
        cv2.putText(frame, f"Calibrating... {int(CALIBRATION_TIME - elapsed)}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        if elapsed >= CALIBRATION_TIME:
            head_calibrated = True
            eye_calibrated = True
            print("✅ Calibration Done")
        cv2.imshow("Proctoring", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    if face_present:
        landmarks = mp_results.multi_face_landmarks[0].landmark
        # Simple eye direction estimation
        left_iris_x = int(landmarks[468].x * w)
        right_iris_x = int(landmarks[473].x * w)
        cv2.putText(frame, f"Left Eye X: {left_iris_x}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Right Eye X: {right_iris_x}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Lips detection
        top_lip = int(landmarks[13].y * h)
        bottom_lip = int(landmarks[14].y * h)
        speaking = abs(top_lip - bottom_lip) > 10
        if speaking:
            log_event("Speaking Detected", frame)
        cv2.putText(frame, f"Speaking: {'Yes' if speaking else 'No'}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if speaking else (0, 255, 0), 2)

    # Multi-person warning
    if person_count > 1:
        log_event("Multiple Persons Detected", frame)
        cv2.putText(frame, "Multiple Persons Detected!", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display
    cv2.imshow("Proctoring", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
