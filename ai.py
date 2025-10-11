import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import csv
import time
from datetime import datetime

# --- Setup ---
yolo_model = YOLO('yolov8n.pt')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
w, h = 640, 480

# Log file setup
log_file = open("proctoring_log.csv", "w", newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Event", "Details"])

# Counter for empty chair
empty_chair_attempts = 0
last_person_seen = time.time()
ALERT_THRESHOLD = 10  # seconds

# Model points for head pose
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye corner
    (225.0, 170.0, -135.0),      # Right eye corner
    (-150.0, -150.0, -125.0),    # Left mouth
    (150.0, -150.0, -125.0)      # Right mouth
])

def log_event(event, detail=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_writer.writerow([timestamp, event, detail])
    print(f"[LOGGED] {timestamp} - {event}: {detail}")

def get_landmark_point(landmarks, index):
    return int(landmarks[index].x * w), int(landmarks[index].y * h)

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
        label = results[0].names[int(cls)]
        if label == "person" and conf > 0.5:
            person_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)

    # FaceMesh detection
    result = face_mesh.process(rgb)
    person_present = result.multi_face_landmarks is not None

    # ---- EMPTY CHAIR LOGIC ----
    if person_count == 1 and not person_present:
        # Person left the frame
        if time.time() - last_person_seen > ALERT_THRESHOLD:
            empty_chair_attempts += 1
            log_event("Empty Chair", f"Attempt {empty_chair_attempts}")
            last_person_seen = time.time()
    elif person_present:
        last_person_seen = time.time()

    if empty_chair_attempts > 0:
        cv2.putText(frame, f"Warning! Person not detected", (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        if empty_chair_attempts >= 2:
            cv2.putText(frame, "Final Warning! Candidate may be disqualified", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    if person_count > 1:
        log_event("Multiple Persons Detected", "Cheating attempt")
        cv2.putText(frame, "Multiple People Detected!", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # If face is detected, do advanced tracking
    if person_present:
        landmarks = result.multi_face_landmarks[0].landmark

        # === Eye Tracking ===
        left_outer = get_landmark_point(landmarks, 33)
        left_inner = get_landmark_point(landmarks, 133)
        left_iris = get_landmark_point(landmarks, 468)

        right_outer = get_landmark_point(landmarks, 362)
        right_inner = get_landmark_point(landmarks, 263)
        right_iris = get_landmark_point(landmarks, 473)

        def get_eye_direction(iris, inner, outer):
            center_x = (inner[0] + outer[0]) // 2
            if iris[0] < center_x - 5:
                return "Left"
            elif iris[0] > center_x + 5:
                return "Right"
            else:
                return "Center"

        left_dir = get_eye_direction(left_iris, left_inner, left_outer)
        right_dir = get_eye_direction(right_iris, right_inner, right_outer)

        cv2.putText(frame, f"Left Eye: {left_dir}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Right Eye: {right_dir}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # === Lips Tracking ===
        top_lip = get_landmark_point(landmarks, 14)
        bottom_lip = get_landmark_point(landmarks, 13)
        lip_distance = abs(top_lip[1] - bottom_lip[1])
        speaking = lip_distance > 10

        if speaking:
            log_event("Speaking Detected", "Possible cheating")
        cv2.putText(frame, f"Speaking: {'Yes' if speaking else 'No'}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255) if speaking else (0, 255, 0), 2)

        # === Head Pose Estimation ===
        image_points = np.array([
            get_landmark_point(landmarks, 1),    # Nose
            get_landmark_point(landmarks, 152),  # Chin
            get_landmark_point(landmarks, 263),  # Right eye
            get_landmark_point(landmarks, 33),   # Left eye
            get_landmark_point(landmarks, 287),  # Right mouth
            get_landmark_point(landmarks, 57),   # Left mouth
        ], dtype='double')

        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4,1))

        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Visualize head pose
        nose_tip = image_points[0]
        nose_end_point_3d = np.array([[0, 0, 1000.0]])
        nose_end_point_2d, _ = cv2.projectPoints(nose_end_point_3d, rotation_vec, translation_vec, camera_matrix, dist_coeffs)

        p1 = (int(nose_tip[0]), int(nose_tip[1]))
        p2 = (int(nose_end_point_2d[0][0][0]), int(nose_end_point_2d[0][0][1]))
        cv2.line(frame, p1, p2, (255, 0, 255), 3)

        # Estimate direction
        rvec_matrix = cv2.Rodrigues(rotation_vec)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        yaw = euler_angles[1]
        pitch = euler_angles[0]

        direction = "Straight"
        if yaw > 15:
            direction = "Right"
        elif yaw < -15:
            direction = "Left"
        elif pitch > 15:
            direction = "Down"
        elif pitch < -15:
            direction = "Up"

        if direction != "Straight":
            log_event("Looking Away", direction)

        cv2.putText(frame, f"Head: {direction}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # Display window
    cv2.imshow("Proctoring AI", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
log_file.close()
cv2.destroyAllWindows()
