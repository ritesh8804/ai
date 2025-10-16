import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import csv
import time
from datetime import datetime
from collections import defaultdict, deque

# ======================
# --- CONFIGURABLES ---
# ======================
CAM_WIDTH, CAM_HEIGHT = 640, 480
CALIBRATION_SECONDS = 3.0
ALERT_THRESHOLD_EMPTY = 10.0  # seconds without YOLO detection => "Empty Chair"
LOG_FILENAME = "proctoring_log.csv"

# ViolationDetector tuning
YAW_THRESH = 18.0
PITCH_THRESH = 18.0
EMA_ALPHA = 0.18
SUSTAIN_TIME = 2.0
MIN_LOG_INTERVAL = 15.0
LIP_SUSTAIN_FRAMES = 10
EYE_AGREEMENT_REQUIRED = False

# Lip threshold (pixels) - may require small tuning per camera
LIP_OPEN_THRESHOLD = 10


# ======================
# --- HELPER CLASSES ---
# ======================
class ViolationDetector:
    def __init__(self,
                 yaw_thresh=YAW_THRESH,
                 pitch_thresh=PITCH_THRESH,
                 ema_alpha=EMA_ALPHA,
                 sustain_time=SUSTAIN_TIME,
                 min_log_interval=MIN_LOG_INTERVAL,
                 lip_sustain_frames=LIP_SUSTAIN_FRAMES,
                 eye_agreement_required=EYE_AGREEMENT_REQUIRED):
        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh
        self.ema_alpha = ema_alpha
        self.yaw_ema = 0.0
        self.pitch_ema = 0.0
        self.sustain_time = sustain_time
        self.min_log_interval = min_log_interval
        self.last_logged = defaultdict(lambda: 0.0)
        self.current_violation_start = {}
        self.lip_queue = deque(maxlen=lip_sustain_frames)
        self.eye_agreement_required = eye_agreement_required

    def _smooth(self, old, new):
        return self.ema_alpha * new + (1 - self.ema_alpha) * old

    def update_head(self, yaw, pitch):
        # initialize EMA on first call
        if self.yaw_ema == 0 and self.pitch_ema == 0:
            self.yaw_ema, self.pitch_ema = yaw, pitch
        else:
            self.yaw_ema = self._smooth(self.yaw_ema, yaw)
            self.pitch_ema = self._smooth(self.pitch_ema, pitch)

        direction = "Straight"
        if self.yaw_ema > self.yaw_thresh:
            direction = "Right"
        elif self.yaw_ema < -self.yaw_thresh:
            direction = "Left"
        elif self.pitch_ema > self.pitch_thresh:
            direction = "Down"
        elif self.pitch_ema < -self.pitch_thresh:
            direction = "Up"
        return direction

    def update_speaking(self, speaking_bool):
        self.lip_queue.append(1 if speaking_bool else 0)
        # speaking considered true when >= 60% of buffer frames indicate speaking
        return sum(self.lip_queue) >= (len(self.lip_queue) * 0.6) if len(self.lip_queue) > 0 else False

    def check_and_log(self, event_name, condition_bool, detail, log_fn):
        now = time.time()
        if condition_bool:
            if event_name not in self.current_violation_start:
                self.current_violation_start[event_name] = now
            elapsed = now - self.current_violation_start[event_name]
            if elapsed >= self.sustain_time and (now - self.last_logged[event_name] >= self.min_log_interval):
                log_fn(event_name, detail)
                self.last_logged[event_name] = now
        else:
            # reset timer if not present
            if event_name in self.current_violation_start:
                del self.current_violation_start[event_name]


# ======================
# --- INIT & MODELS ---
# ======================
# Load YOLO and MediaPipe Face Mesh
yolo_model = YOLO('yolov8n.pt')  # ensure model file is available
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# CSV logging setup
log_file = open(LOG_FILENAME, "w", newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Event", "Details"])

# Create violation detector
violation_detector = ViolationDetector()

# Head pose 3D model points (standard face model)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye corner
    (225.0, 170.0, -135.0),      # Right eye corner
    (-150.0, -150.0, -125.0),    # Left mouth
    (150.0, -150.0, -125.0)      # Right mouth
])

# Tracking
empty_chair_attempts = 0
last_person_seen = time.time()

# Calibration storage
calibrating = True
calib_start_time = None
calib_samples = []
baseline_yaw = 0.0
baseline_pitch = 0.0
baseline_initialized = False


# ======================
# --- HELPERS ---
# ======================
def log_event(event, detail=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_writer.writerow([timestamp, event, detail])
    print(f"[LOGGED] {timestamp} - {event}: {detail}")


def get_landmark_point(landmarks, index):
    return int(landmarks[index].x * CAM_WIDTH), int(landmarks[index].y * CAM_HEIGHT)


def draw_center_text(img, text, y, scale=0.8, color=(255, 255, 255), thickness=2):
    (w_text, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max((CAM_WIDTH - w_text) // 2, 10)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ======================
# --- MAIN LOOP ---
# ======================
print("Starting. Press ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO detection (person)
    results = yolo_model(frame)
    detections = results[0].boxes.data
    person_count = 0
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        label = results[0].names[int(cls)]
        if label == "person" and conf > 0.5:
            person_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # FaceMesh detection
    result = face_mesh.process(rgb)
    person_present = result.multi_face_landmarks is not None

    # --- Empty chair logic ---
    if person_count == 0:
        if time.time() - last_person_seen > ALERT_THRESHOLD_EMPTY and (time.time() - violation_detector.last_logged["Empty Chair"] >= violation_detector.min_log_interval):
            log_event("Empty Chair", "No person detected by YOLO for sustained period")
            violation_detector.last_logged["Empty Chair"] = time.time()
    else:
        last_person_seen = time.time()

    # --- Multiple persons detection (cooldowned) ---
    if person_count > 1:
        violation_detector.check_and_log("Multiple Persons Detected", True, f"Count={person_count}", log_event)
    else:
        violation_detector.check_and_log("Multiple Persons Detected", False, "", log_event)

    # --- Calibration phase handling ---
    if calibrating:
        # Start calibration timer on first frame of calibration
        if calib_start_time is None:
            calib_start_time = time.time()
            calib_samples = []
            print("Calibration started. Please sit straight and look at the camera.")

        elapsed = time.time() - calib_start_time
        remaining = max(0.0, CALIBRATION_SECONDS - elapsed)

        draw_center_text(frame, f"Calibration: Please sit straight ({remaining:.1f}s)", CAM_HEIGHT // 2, scale=0.9, color=(0, 255, 255), thickness=2)

        # If face & head pose available, collect yaw/pitch
        if person_present:
            try:
                landmarks = result.multi_face_landmarks[0].landmark
                image_points = np.array([
                    get_landmark_point(landmarks, 1),
                    get_landmark_point(landmarks, 152),
                    get_landmark_point(landmarks, 263),
                    get_landmark_point(landmarks, 33),
                    get_landmark_point(landmarks, 287),
                    get_landmark_point(landmarks, 57),
                ], dtype='double')

                focal_length = CAM_WIDTH
                center = (CAM_WIDTH / 2, CAM_HEIGHT / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype="double")
                dist_coeffs = np.zeros((4, 1))

                success, rotation_vec, translation_vec = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                if success:
                    rvec_matrix = cv2.Rodrigues(rotation_vec)[0]
                    proj_matrix = np.hstack((rvec_matrix, translation_vec))
                    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
                    raw_yaw, raw_pitch = float(euler_angles[1]), float(euler_angles[0])
                    calib_samples.append((raw_yaw, raw_pitch))
            except Exception:
                pass  # ignore occasional solvePnP errors during calibration

        # Finish calibration when time's up
        if elapsed >= CALIBRATION_SECONDS:
            if len(calib_samples) > 0:
                ys = [s[0] for s in calib_samples]
                ps = [s[1] for s in calib_samples]
                baseline_yaw = float(np.mean(ys))
                baseline_pitch = float(np.mean(ps))
                baseline_initialized = True
                print(f"Calibration complete. Baseline yaw={baseline_yaw:.3f}, pitch={baseline_pitch:.3f}")
                log_event("Calibration Complete", f"Baseline yaw={baseline_yaw:.3f}, pitch={baseline_pitch:.3f}")
            else:
                # fallback if no samples collected
                baseline_yaw = 0.0
                baseline_pitch = 0.0
                baseline_initialized = True
                print("Calibration complete but no face samples captured -> using zero baseline.")
                log_event("Calibration Complete", "No samples captured; using zero baseline")
            calibrating = False

        # show frame and skip the main detection state until calibration finishes
        cv2.imshow("AI Proctoring System - Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue  # go to next loop frame while calibrating

    # --- Post-calibration: main monitoring ---
    if person_present:
        try:
            landmarks = result.multi_face_landmarks[0].landmark

            # Eye tracking
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

            # Lips
            top_lip = get_landmark_point(landmarks, 14)
            bottom_lip = get_landmark_point(landmarks, 13)
            lip_distance = abs(top_lip[1] - bottom_lip[1])
            speaking = lip_distance > LIP_OPEN_THRESHOLD
            speaking_sustained = violation_detector.update_speaking(speaking)

            # Head pose via solvePnP
            image_points = np.array([
                get_landmark_point(landmarks, 1),
                get_landmark_point(landmarks, 152),
                get_landmark_point(landmarks, 263),
                get_landmark_point(landmarks, 33),
                get_landmark_point(landmarks, 287),
                get_landmark_point(landmarks, 57),
            ], dtype='double')

            focal_length = CAM_WIDTH
            center = (CAM_WIDTH / 2, CAM_HEIGHT / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            if success:
                rvec_matrix = cv2.Rodrigues(rotation_vec)[0]
                proj_matrix = np.hstack((rvec_matrix, translation_vec))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
                raw_yaw, raw_pitch = float(euler_angles[1]), float(euler_angles[0])

                # subtract calibration baseline so neutral position isn't flagged
                if baseline_initialized:
                    adj_yaw = raw_yaw - baseline_yaw
                    adj_pitch = raw_pitch - baseline_pitch
                else:
                    adj_yaw, adj_pitch = raw_yaw, raw_pitch

                # use the detector that includes EMA smoothing
                direction = violation_detector.update_head(adj_yaw, adj_pitch)

                # eyes away & suspicious composition
                eyes_away = (left_dir != "Center" and right_dir != "Center")
                if violation_detector.eye_agreement_required:
                    suspicious = (direction != "Straight") and eyes_away
                else:
                    # be a bit tolerant: require head away OR both eyes away OR speaking
                    suspicious = (direction != "Straight") or (eyes_away and speaking_sustained)

                # Controlled logging
                violation_detector.check_and_log("Looking Away", suspicious, f"Head={direction}, Yaw={adj_yaw:.2f}, Pitch={adj_pitch:.2f}, Eyes={left_dir}/{right_dir}", log_event)
                violation_detector.check_and_log("Speaking Detected", speaking_sustained, f"LipDistance={lip_distance}", log_event)

                # Visual: draw nose direction line for user feedback
                nose_tip = image_points[0]
                nose_end_point_3d = np.array([[0, 0, 1000.0]])
                nose_end_point_2d, _ = cv2.projectPoints(nose_end_point_3d, rotation_vec, translation_vec, camera_matrix, dist_coeffs)
                p1 = (int(nose_tip[0]), int(nose_tip[1]))
                p2 = (int(nose_end_point_2d[0][0][0]), int(nose_end_point_2d[0][0][1]))
                cv2.line(frame, p1, p2, (255, 0, 255), 2)
            else:
                # solvePnP occasionally fails; do not treat as violation
                direction = "Unknown"
                adj_yaw = adj_pitch = 0.0

            # Draw HUD
            cv2.putText(frame, f"Head: {direction}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.putText(frame, f"Yaw: {adj_yaw:.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Pitch: {adj_pitch:.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Left Eye: {left_dir}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Right Eye: {right_dir}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Speaking: {'Yes' if speaking_sustained else 'No'}", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255) if speaking_sustained else (0, 255, 0), 1)

        except Exception as e:
            # avoid the program crashing on occasional landmarks/pose errors
            # print(e)  # uncomment for debugging
            pass
    else:
        # No face found: display warning but don't immediately log; empty chair logic handles sustained absence
        cv2.putText(frame, "No face detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("AI Proctoring System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    # optional: allow manual recalibration with 'c'
    if key == ord('c'):
        calibrating = True
        calib_start_time = None
        calib_samples = []
        baseline_initialized = False
        print("Manual recalibration triggered.")

# Cleanup
cap.release()
log_file.close()
cv2.destroyAllWindows()
print(f"Session ended. Logs saved to {LOG_FILENAME}")
