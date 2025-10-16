import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from deepface import DeepFace
import csv
import time
from datetime import datetime

# --- Setup ---
yolo_model = YOLO('yolov8n.pt')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
w, h = 640, 480

# Log file setup
log_file = open("proctoring_log.csv", "w", newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Event", "Details"])

# Empty chair & head pose counters
empty_chair_attempts = 0
last_person_seen = time.time()
ALERT_THRESHOLD = 10  # seconds

# Calibration
head_calibrated = False
eye_calibrated = False
head_pitch_offset = 0
head_yaw_offset = 0
left_eye_center_offset = 0
right_eye_center_offset = 0
CALIBRATION_TIME = 3  # seconds
calibration_start_time = None

# Identity verification
REFERENCE_IMAGE_PATH = "reference.jpg"  # local reference image
identity_threshold = 0.90
unrecognized_start = None
sustain_time_identity = 2.0
reference_embedding = DeepFace.represent(img_path=REFERENCE_IMAGE_PATH, model_name='Facenet')[0]["embedding"]

# Head pose model points
model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
])

def log_event(event, detail=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_writer.writerow([timestamp, event, detail])
    print(f"[LOGGED] {timestamp} - {event}: {detail}")

def get_landmark_point(landmarks, index):
    return int(landmarks[index].x * w), int(landmarks[index].y * h)

def get_eye_direction(iris, inner, outer, baseline_offset):
    center_x = (inner[0] + outer[0]) // 2 + baseline_offset
    if iris[0] < center_x - 5:
        return "Left"
    elif iris[0] > center_x + 5:
        return "Right"
    else:
        return "Center"

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (w, h))
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO detection
    results = yolo_model(frame)
    detections = results[0].boxes.data
    person_count = 0
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if results[0].names[int(cls)] == "person" and conf > 0.5:
            person_count += 1

    # FaceMesh detection
    mp_results = face_mesh.process(rgb)
    face_present = mp_results.multi_face_landmarks is not None

    # --- EMPTY CHAIR LOGIC ---
    if person_count == 1 and not face_present:
        if time.time() - last_person_seen > ALERT_THRESHOLD:
            empty_chair_attempts += 1
            log_event("Empty Chair", f"Attempt {empty_chair_attempts}")
            last_person_seen = time.time()
    elif face_present:
        last_person_seen = time.time()

    # --- Multiple person warning ---
    if person_count > 1:
        log_event("Multiple Persons Detected", "Cheating attempt")
        cv2.putText(frame, "Multiple People Detected!", (20, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # --- Face detected: Calibration + Head/Eye + Identity ---
    if face_present:
        landmarks = mp_results.multi_face_landmarks[0].landmark

        # Calibration phase
        if not head_calibrated or not eye_calibrated:
            if calibration_start_time is None:
                calibration_start_time = time.time()
            elapsed = time.time() - calibration_start_time
            cv2.putText(frame, f"Calibration... {int(CALIBRATION_TIME - elapsed)}s", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            if elapsed >= CALIBRATION_TIME:
                # Head pose baseline
                image_points = np.array([
                    get_landmark_point(landmarks, 1),
                    get_landmark_point(landmarks, 152),
                    get_landmark_point(landmarks, 263),
                    get_landmark_point(landmarks, 33),
                    get_landmark_point(landmarks, 287),
                    get_landmark_point(landmarks, 57),
                ], dtype='double')
                focal_length = w
                center = (w/2, h/2)
                camera_matrix = np.array([[focal_length, 0, center[0]],
                                          [0, focal_length, center[1]],
                                          [0, 0, 1]], dtype="double")
                dist_coeffs = np.zeros((4,1))
                success, rotation_vec, translation_vec = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                rvec_matrix = cv2.Rodrigues(rotation_vec)[0]
                proj_matrix = np.hstack((rvec_matrix, translation_vec))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
                head_pitch_offset = euler_angles[0]
                head_yaw_offset = euler_angles[1]

                # Eye baseline
                left_inner = get_landmark_point(landmarks, 133)
                left_outer = get_landmark_point(landmarks, 33)
                left_iris = get_landmark_point(landmarks, 468)
                left_eye_center_offset = (left_iris[0] - (left_inner[0] + left_outer[0])//2)

                right_inner = get_landmark_point(landmarks, 362)
                right_outer = get_landmark_point(landmarks, 263)
                right_iris = get_landmark_point(landmarks, 473)
                right_eye_center_offset = (right_iris[0] - (right_inner[0] + right_outer[0])//2)

                head_calibrated = True
                eye_calibrated = True
            continue  # skip rest until calibrated

        # Eye tracking (mirrored correction)
        left_outer = get_landmark_point(landmarks, 362)
        left_inner = get_landmark_point(landmarks, 263)
        left_iris = get_landmark_point(landmarks, 473)

        right_outer = get_landmark_point(landmarks, 33)
        right_inner = get_landmark_point(landmarks, 133)
        right_iris = get_landmark_point(landmarks, 468)

        left_dir = get_eye_direction(left_iris, left_inner, left_outer, left_eye_center_offset)
        right_dir = get_eye_direction(right_iris, right_inner, right_outer, right_eye_center_offset)

        cv2.putText(frame, f"Left Eye: {left_dir}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
        cv2.putText(frame, f"Right Eye: {right_dir}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)

        # Identity verification (compare embedding with reference)
        xs = [int(l.x*w) for l in landmarks]
        ys = [int(l.y*h) for l in landmarks]
        x_min, x_max = max(0, min(xs)), min(w, max(xs))
        y_min, y_max = max(0, min(ys)), min(h, max(ys))
        face_crop = frame[y_min:y_max, x_min:x_max]

        try:
            emb = DeepFace.represent(face_crop, model_name='Facenet', enforce_detection=False)[0]["embedding"]
            similarity = np.dot(reference_embedding, emb) / (np.linalg.norm(reference_embedding)*np.linalg.norm(emb))
            recognized = similarity >= identity_threshold
            if not recognized:
                now = time.time()
                if unrecognized_start is None:
                    unrecognized_start = now
                elif now - unrecognized_start >= sustain_time_identity:
                    log_event("Unrecognized Person", f"Similarity={similarity:.2f}")
                    unrecognized_start = now
            else:
                unrecognized_start = None

            cv2.putText(frame, f"Identity: {'OK' if recognized else 'Unknown'}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if recognized else (0,0,255),2)
        except Exception as e:
            print("DeepFace error:", e)

    cv2.imshow("AI Proctoring System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
