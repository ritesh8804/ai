"""
Enhanced Real-time Proctoring System (False-Positive Resistant Speech Detection)

What’s new vs prior version:
- Higher thresholds to reduce sensitivity (outer/inner MAR, lip distance)
- Temporal persistence: require 6 of last 10 frames to be positive
- EMA smoothing for MAR metrics to reduce jitter
- Smile-guard: wide mouth width without vertical opening does not count as speaking
- Optional per-user baseline during first 60 stable frames for adaptive thresholds
- Keeps all previous features (YOLO person/phone/chair, head pose, eye gaze, hand cover)
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
from collections import deque
import os
from datetime import datetime
import csv
import html

# === CONFIG ===
YOLO_MODEL_PATH = "yolov8n.pt"
CAM_ID = 0
FRAME_W = 1280
FRAME_H = 720

# YOLO thresholds
PERSON_CONF = 0.35
OTHER_CONF = 0.35
IOU_EMPTY_CHAIR = 0.25
PHONE_CONF = 0.35
MULTI_PERSON_COUNT = 2

# Head movement thresholds
HEAD_MOVE_YAW_THRESHOLD = 25.0
HEAD_MOVEMENT_VARIANCE_WINDOW = 20
HEAD_MOVEMENT_STD_THRESHOLD = 8.0

# Speech detection thresholds (tuned to reduce false positives)
MAR_SPEAK_THRESHOLD_OUTER_BASE = 0.28   # was 0.25
MAR_SPEAK_THRESHOLD_INNER_BASE = 0.18   # was 0.15
LIP_DISTANCE_THRESHOLD_BASE = 8.0       # was 6.0 (pixels)

# Temporal smoothing and persistence
MAR_MOVEMENT_WINDOW = 20                # was 15
MAR_MOVEMENT_STD_OUTER = 0.015          # was 0.012
MAR_MOVEMENT_STD_INNER = 0.010          # was 0.008
SPEAK_HIST_LEN = 10                      # frames
SPEAK_MIN_COUNT = 6                      # require >=6 positives in last 10

# Decision rule tightening
CONFIDENCE_MIN = 0.45                    # require combined conf > 0.45
REQUIRE_TWO_VOTES = True                 # require 2+ votes

# Smile-guard (mouth width big, vertical small => likely smile)
SMILE_WIDTH_GAIN = 1.08                  # width increase threshold vs baseline
SMILE_MAX_VDIST = 7.0                    # px maximum vertical to still be considered smile

# Baseline collection (adaptive thresholds)
ENABLE_BASELINE = True
BASELINE_FRAMES = 60                     # initial stable frames to collect baseline
BASELINE_MAX_MOTION_STD = 0.010          # only collect when inner MAR std small
BASELINE_OUTER_ADD = 0.06                # add on top of baseline
BASELINE_INNER_ADD = 0.04
BASELINE_VDIST_ADD = 3.0                 # px add

# Eye gaze
EYE_GAZE_THRESHOLD = 0.28

# Visualization
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Logging / dashboard
LOG_DIR = "proctor_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "cheating_log.txt")
CSV_FILE = os.path.join(LOG_DIR, "cheating_log.csv")
DASHBOARD_FILE = os.path.join(LOG_DIR, "dashboard.html")
MAX_DASH_EVENTS = 50
LOG_COOLDOWN_SEC = 1.0

# CSV init
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "timestamp", "reason", "evidence_path",
            "persons", "phones", "chairs",
            "mar_outer", "mar_inner", "lip_distance", "speech_confidence"
        ])

def update_dashboard():
    rows = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", encoding="utf-8") as cf:
            reader = csv.reader(cf)
            header = next(reader, None)
            for r in reader:
                rows.append(r)
    rows = rows[-MAX_DASH_EVENTS:][::-1]

    html_rows = ""
    for r in rows:
        if len(r) >= 10:
            ts, reason, img_name = r[0], r[1], r[2]
            persons, phones, chairs = r[3], r[4], r[5]
            mar_outer, mar_inner, lip_dist, speech_conf = r[6], r[7], r[8], r[9]
            safe_reason = html.escape(reason)
            html_rows += f"""
            <tr>
                <td>{ts}</td>
                <td>{safe_reason}</td>
                <td><img src="{img_name}" alt="evidence" style="max-width:140px; max-height:100px;" /></td>
                <td>{persons}</td><td>{phones}</td><td>{chairs}</td>
                <td>{mar_outer}</td><td>{mar_inner}</td><td>{lip_dist}</td><td>{speech_conf}</td>
            </tr>
            """
        else:
            ts, reason, img_name = r[0], r[1], r[2]
            persons = r[3] if len(r) > 3 else "0"
            phones = r[4] if len(r) > 4 else "0"
            chairs = r[5] if len(r) > 5 else "0"
            safe_reason = html.escape(reason)
            html_rows += f"""
            <tr>
                <td>{ts}</td>
                <td>{safe_reason}</td>
                <td><img src="{img_name}" alt="evidence" style="max-width:140px; max-height:100px;" /></td>
                <td>{persons}</td><td>{phones}</td><td>{chairs}</td>
                <td colspan="4">Legacy event</td>
            </tr>
            """

    html_content = f"""
    <html>
    <head><meta charset="utf-8"><title>Enhanced Proctoring Dashboard</title></head>
    <body>
    <h2>Enhanced Proctoring - Recent Events (latest first)</h2>
    <table border="1" cellpadding="5" cellspacing="0">
        <thead>
            <tr><th>Timestamp</th><th>Reason</th><th>Evidence</th><th>Persons</th><th>Phones</th><th>Chairs</th><th>MAR_out</th><th>MAR_in</th><th>Lip_dist</th><th>Speech_conf</th></tr>
        </thead>
        <tbody>
        {html_rows if html_rows else '<tr><td colspan="10">No events logged yet</td></tr>'}
        </tbody>
    </table>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </body></html>
    """
    with open(DASHBOARD_FILE, "w", encoding="utf-8") as hf:
        hf.write(html_content)

def log_event(reason, frame, persons_count=0, phones_count=0, chairs_count=0,
              mar_outer=0.0, mar_inner=0.0, lip_distance=0.0, speech_confidence=0.0):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_name = f"evidence_{ts}.jpg"
    img_path = os.path.join(LOG_DIR, img_name)
    cv2.imwrite(img_path, frame)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] CHEATING: {reason} | Evidence: {img_path} | "
                f"Persons:{persons_count} Phones:{phones_count} Chairs:{chairs_count} | "
                f"MAR_out:{mar_outer:.3f} MAR_in:{mar_inner:.3f} Lip_dist:{lip_distance:.1f} "
                f"Speech_conf:{speech_confidence:.3f}\n")
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            ts, reason, img_name, persons_count, phones_count, chairs_count,
            f"{mar_outer:.3f}", f"{mar_inner:.3f}", f"{lip_distance:.1f}", f"{speech_confidence:.3f}"
        ])
    try:
        update_dashboard()
    except Exception as e:
        print("[WARN] update_dashboard failed:", e)
    print(f"[LOGGED] {reason} @ {ts} | MAR_out:{mar_outer:.3f} MAR_in:{mar_inner:.3f} Dist:{lip_distance:.1f} Conf:{speech_confidence:.2f}")

# === MediaPipe and YOLO init ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=2,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.7)  # slightly higher tracking conf to reduce jitter

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

yolo = YOLO(YOLO_MODEL_PATH)

# Landmark indices
IDX = {
    'nose_tip': 1, 'chin': 152,
    'left_eye_outer': 33, 'right_eye_outer': 263,
    'mouth_left': 61, 'mouth_right': 291,
    'left_iris': 468, 'right_iris': 473,
    'upper_lip_center': 13, 'lower_lip_center': 14,
    'inner_upper': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    'inner_lower': [324, 318, 402, 317, 14, 87, 178, 88, 95],
    'outer_upper': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    'outer_lower': [146, 91, 181, 84, 17, 314, 405, 321, 375],
}

# 3D model points for head pose
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float64)

def bbox_iou(boxA, boxB):
    xa = max(boxA[0], boxB[0]); ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2]); yb = min(boxA[3], boxB[3])
    inter_w = max(0, xb - xa); inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return 0.0 if union == 0 else inter/union

def mediapipe_landmarks_to_np(landmarks, img_w, img_h):
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * img_w), int(lm.y * img_h), lm.z))
    return np.array(pts)

def estimate_head_pose(pts2d, camera_matrix, dist_coeffs=np.zeros((4,1))):
    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_3D_POINTS, pts2d.astype(np.float64), camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None
    rmat, _ = cv2.Rodrigues(rotation_vec)
    proj_matrix = np.hstack((rmat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = float(euler_angles[0]), float(euler_angles[1]), float(euler_angles[2])
    return (yaw, pitch, roll)

def eye_iris_offset(eye_outer, eye_inner, iris_center):
    ex, ey = eye_outer; ix, iy = eye_inner; cx, cy = iris_center
    bx = (ex + ix) / 2; by = (ey + iy) / 2
    w = abs(ex - ix); h = w * 0.6 if w>0 else 1.0
    dx = (cx - bx) / (w/2 + 1e-6)
    dy = (cy - by) / (h/2 + 1e-6)
    return dx, dy

def mouth_width(pts):
    left_corner = pts[IDX['mouth_left']][:2]
    right_corner = pts[IDX['mouth_right']][:2]
    return float(np.linalg.norm(np.array(left_corner) - np.array(right_corner)))

def mar_outer(pts):
    try:
        upper_pts = [pts[i][:2] for i in IDX['outer_upper'][:5]]
        lower_pts = [pts[i][:2] for i in IDX['outer_lower'][:5]]
        vds = [np.linalg.norm(np.array(up) - np.array(lp)) for up, lp in zip(upper_pts, lower_pts)]
        width = mouth_width(pts)
        if width < 1e-6: return 0.0
        return float(np.mean(vds)/width)
    except Exception:
        return 0.0

def mar_inner(pts):
    try:
        upper_pts = [pts[i][:2] for i in IDX['inner_upper'][:5]]
        lower_pts = [pts[i][:2] for i in IDX['inner_lower'][:5]]
        vds = [np.linalg.norm(np.array(up) - np.array(lp)) for up, lp in zip(upper_pts, lower_pts)]
        inner_left = pts[78][:2]; inner_right = pts[308][:2]
        inner_width = np.linalg.norm(np.array(inner_left) - np.array(inner_right))
        if inner_width < 1e-6: return 0.0
        return float(np.mean(vds)/(inner_width + 1e-6))
    except Exception:
        return 0.0

def lips_vertical_distance(pts):
    try:
        top_center = pts[IDX['upper_lip_center']][:2]
        bottom_center = pts[IDX['lower_lip_center']][:2]
        return float(np.linalg.norm(np.array(top_center) - np.array(bottom_center)))
    except Exception:
        return 0.0

class SpeechState:
    def __init__(self):
        self.mar_out_buf = deque(maxlen=MAR_MOVEMENT_WINDOW)
        self.mar_in_buf = deque(maxlen=MAR_MOVEMENT_WINDOW)
        self.speak_hist = deque(maxlen=SPEAK_HIST_LEN)
        self.mar_out_ema = None
        self.mar_in_ema = None
        self.width_baseline = None
        self.base_collect = []
        self.base_ready = False
        self.thr_out = MAR_SPEAK_THRESHOLD_OUTER_BASE
        self.thr_in = MAR_SPEAK_THRESHOLD_INNER_BASE
        self.thr_dist = LIP_DISTANCE_THRESHOLD_BASE

    def update_baseline(self, m_out, m_in, vdist):
        if not ENABLE_BASELINE or self.base_ready:
            return
        # collect only when motion is low (use inner buffer std as proxy)
        self.mar_in_buf.append(m_in)
        if len(self.mar_in_buf) >= 10:
            if np.std(list(self.mar_in_buf)[-10:]) < BASELINE_MAX_MOTION_STD:
                self.base_collect.append((m_out, m_in, vdist))
        if len(self.base_collect) >= BASELINE_FRAMES:
            data = np.array(self.base_collect)
            base_out = float(np.median(data[:,0]))
            base_in = float(np.median(data[:,1]))
            base_vd = float(np.median(data[:,2]))
            self.thr_out = max(MAR_SPEAK_THRESHOLD_OUTER_BASE, base_out + BASELINE_OUTER_ADD)
            self.thr_in = max(MAR_SPEAK_THRESHOLD_INNER_BASE, base_in + BASELINE_INNER_ADD)
            self.thr_dist = max(LIP_DISTANCE_THRESHOLD_BASE, base_vd + BASELINE_VDIST_ADD)
            self.base_ready = True

    def smooth(self, val, prev, alpha=0.3):
        return val if prev is None else (alpha*val + (1-alpha)*prev)

    def decide(self, pts, m_out, m_in, vdist):
        # EMA smoothing to reduce jitter
        self.mar_out_ema = self.smooth(m_out, self.mar_out_ema, alpha=0.3)
        self.mar_in_ema = self.smooth(m_in, self.mar_in_ema, alpha=0.3)
        mo = self.mar_out_ema if self.mar_out_ema is not None else m_out
        mi = self.mar_in_ema if self.mar_in_ema is not None else m_in

        self.mar_out_buf.append(mo)
        self.mar_in_buf.append(mi)

        # thresholds (adaptive if baseline ready)
        thr_out = self.thr_out
        thr_in = self.thr_in
        thr_dist = self.thr_dist

        # primary votes
        v_out = (mo > thr_out)
        v_in = (mi > thr_in)
        v_dist = (vdist > thr_dist)

        # movement votes
        v_mov_out = False
        v_mov_in = False
        if len(self.mar_out_buf) > 5:
            v_mov_out = np.std(list(self.mar_out_buf)[-10:]) > MAR_MOVEMENT_STD_OUTER
        if len(self.mar_in_buf) > 5:
            v_mov_in = np.std(list(self.mar_in_buf)[-10:]) > MAR_MOVEMENT_STD_INNER

        # confidence
        conf_parts = []
        if mo > 0: conf_parts.append(min(mo / thr_out, 2.0) * 0.30)
        if mi > 0: conf_parts.append(min(mi / thr_in, 2.0) * 0.40)
        if vdist > 0: conf_parts.append(min(vdist / thr_dist, 2.0) * 0.20)
        if v_mov_out: conf_parts.append(0.05)
        if v_mov_in: conf_parts.append(0.15)
        confidence = min(sum(conf_parts), 1.0)

        # smile-guard: if width expands but vertical is small, suppress
        width = mouth_width(pts)
        if self.width_baseline is None:
            self.width_baseline = width
        width_gain = (width / max(self.width_baseline, 1e-6)) if self.width_baseline else 1.0
        if width_gain > SMILE_WIDTH_GAIN and vdist < SMILE_MAX_VDIST:
            v_out = False
            v_in = False
            v_dist = False

        votes = [v_out, v_in, v_dist, v_mov_out, v_mov_in]
        if REQUIRE_TWO_VOTES:
            speaking = (sum(votes) >= 2) and (confidence > CONFIDENCE_MIN)
        else:
            speaking = (sum(votes) >= 1) and (confidence > CONFIDENCE_MIN)

        # temporal persistence
        self.speak_hist.append(1 if speaking else 0)
        speaking_persist = (sum(self.speak_hist) >= SPEAK_MIN_COUNT)

        metrics = {
            "mar_outer": mo,
            "mar_inner": mi,
            "lip_distance": vdist,
            "outer_speaking": v_out,
            "inner_speaking": v_in,
            "distance_speaking": v_dist,
            "movement_speaking": v_mov_out or v_mov_in,
            "votes": int(sum(votes)),
            "width_gain": width_gain,
            "thr_out": thr_out, "thr_in": thr_in, "thr_dist": thr_dist,
        }
        return speaking_persist, confidence, metrics

# Per-face state map (simple single-face assumption -> use one state)
speech_state = SpeechState()

# Buffers
head_yaw_buffer = deque(maxlen=HEAD_MOVEMENT_VARIANCE_WINDOW)
last_log_time = 0

# Camera
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

focal_length = FRAME_W
center = (FRAME_W/2, FRAME_H/2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")

print("[INFO] Starting enhanced proctoring. Press 'q' to quit.")
print(f"[INFO] Base thresholds -> Outer:{MAR_SPEAK_THRESHOLD_OUTER_BASE} Inner:{MAR_SPEAK_THRESHOLD_INNER_BASE} Dist(px):{LIP_DISTANCE_THRESHOLD_BASE}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_h, img_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO detection
    yolo_results = yolo.predict(frame, imgsz=640, conf=PERSON_CONF, verbose=False)[0]
    detected_persons, detected_phones, detected_chairs = [], [], []
    for box, conf, cls in zip(yolo_results.boxes.xyxy.tolist(),
                              yolo_results.boxes.conf.tolist(),
                              yolo_results.boxes.cls.tolist()):
        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)
        name = yolo.model.names.get(cls, str(cls))
        if name == 'person' and conf >= PERSON_CONF:
            detected_persons.append((x1, y1, x2, y2, conf))
        elif name in ('cell phone', 'cellphone', 'phone') and conf >= PHONE_CONF:
            detected_phones.append((x1, y1, x2, y2, conf))
        elif name == 'chair' and conf >= OTHER_CONF:
            detected_chairs.append((x1, y1, x2, y2, conf))

    multi_person_flag = len(detected_persons) >= MULTI_PERSON_COUNT

    # Empty chair
    empty_chair_flag = False
    for chair in detected_chairs:
        cbox = chair[:4]
        overlapped = any(bbox_iou(cbox, p[:4]) > IOU_EMPTY_CHAIR for p in detected_persons)
        if not overlapped:
            empty_chair_flag = True
            cv2.rectangle(frame, (cbox[0], cbox[1]), (cbox[2], cbox[3]), (0,165,255), 2)
            cv2.putText(frame, "EMPTY CHAIR", (cbox[0], cbox[1]-8), FONT, 0.5, (0,165,255), 2)

    phone_flag = len(detected_phones) > 0
    for ph in detected_phones:
        x1,y1,x2,y2,_ = ph
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(frame, "PHONE", (x1, y1-8), FONT, 0.6, (0,0,255), 2)

    for p in detected_persons:
        x1,y1,x2,y2,conf = p
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-8), FONT, 0.5, (0,255,0), 1)

    faces = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    # Hand landmarks
    hand_landmark_points = []
    if hand_results.multi_hand_landmarks:
        for hland in hand_results.multi_hand_landmarks:
            for hl in hland.landmark:
                hx, hy = int(hl.x * img_w), int(hl.y * img_h)
                hand_landmark_points.append((hx, hy))
        for (hx,hy) in hand_landmark_points:
            cv2.circle(frame, (hx,hy), 2, (0,120,255), -1)

    face_flags = []
    if faces.multi_face_landmarks:
        for face_landmarks in faces.multi_face_landmarks:
            lm = face_landmarks.landmark
            pts = mediapipe_landmarks_to_np(lm, img_w, img_h)

            # Head pose
            try:
                p2d = np.array([
                    (pts[IDX['nose_tip']][0], pts[IDX['nose_tip']][1]),
                    (pts[IDX['chin']][0], pts[IDX['chin']][1]),
                    (pts[IDX['left_eye_outer']][0], pts[IDX['left_eye_outer']][1]),
                    (pts[IDX['right_eye_outer']][0], pts[IDX['right_eye_outer']][1]),
                    (pts[IDX['mouth_left']][0], pts[IDX['mouth_left']][1]),
                    (pts[IDX['mouth_right']][0], pts[IDX['mouth_right']][1]),
                ], dtype=np.float64)
            except Exception:
                continue
            head_pose = estimate_head_pose(p2d, camera_matrix)
            yaw, pitch, roll = (0,0,0)
            if head_pose is not None:
                yaw, pitch, roll = head_pose
                head_yaw_buffer.append(yaw)

            # Nose direction viz
            nose_2d = (int(pts[IDX['nose_tip']][0]), int(pts[IDX['nose_tip']][1]))
            try:
                success, rvec, tvec = cv2.solvePnP(MODEL_3D_POINTS, p2d, camera_matrix, np.zeros((4,1)),
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
                if success:
                    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]),
                                                              rvec, tvec, camera_matrix, np.zeros((4,1)))
                    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                    cv2.line(frame, nose_2d, p2, (255,0,0), 2)
            except Exception:
                pass

            # Eye gaze
            left_outer = (pts[33][0], pts[33][1])
            right_outer = (pts[263][0], pts[263][1])
            left_iris = (int(pts[IDX['left_iris']][0]), int(pts[IDX['left_iris']][1]))
            right_iris = (int(pts[IDX['right_iris']][0]), int(pts[IDX['right_iris']][1]))
            left_offset = eye_iris_offset(left_outer, pts[133][:2], left_iris)
            right_offset = eye_iris_offset(pts[362][:2], right_outer, right_iris)
            gaze_side = abs(left_offset[0]) > EYE_GAZE_THRESHOLD or abs(right_offset[0]) > EYE_GAZE_THRESHOLD

            # Speech detection
            m_out = mar_outer(pts)
            m_in = mar_inner(pts)
            vdist = lips_vertical_distance(pts)

            # Update baseline adaptively in first seconds when motion is low
            speech_state.update_baseline(m_out, m_in, vdist)

            speaking_persist, speech_confidence, speech_metrics = speech_state.decide(pts, m_out, m_in, vdist)

            # Head movement
            head_move_flag = False
            if len(head_yaw_buffer) >= 5:
                std_yaw = np.std(head_yaw_buffer)
                head_move_flag = std_yaw > HEAD_MOVEMENT_STD_THRESHOLD

            # Hand covering
            hand_cover = False
            if len(hand_landmark_points) > 0:
                nose_xy = (int(lm[IDX['nose_tip']].x * img_w), int(lm[IDX['nose_tip']].y * img_h))
                mouth_left_xy = (int(lm[IDX['mouth_left']].x * img_w), int(lm[IDX['mouth_left']].y * img_h))
                mouth_right_xy = (int(lm[IDX['mouth_right']].x * img_w), int(lm[IDX['mouth_right']].y * img_h))
                mouth_center = ((mouth_left_xy[0]+mouth_right_xy[0])//2, (mouth_left_xy[1]+mouth_right_xy[1])//2)
                for (hx,hy) in hand_landmark_points:
                    if abs(hx - nose_xy[0]) < 70 and abs(hy - nose_xy[1]) < 70: hand_cover = True
                    if abs(hx - mouth_center[0]) < 80 and abs(hy - mouth_center[1]) < 80: hand_cover = True

            face_flags.append({
                "yaw": yaw, "pitch": pitch, "roll": roll,
                "gaze_side": gaze_side,
                "speaking": speaking_persist,
                "speech_confidence": speech_confidence,
                "speech_metrics": speech_metrics,
                "head_move": head_move_flag,
                "hand_covering": hand_cover,
                "nose_2d": nose_2d,
                "landmarks": pts
            })

            # Visuals
            for (x,y,_) in pts[[33,263,61,291,1,152]]:
                cv2.circle(frame, (int(x),int(y)), 1, (255,255,0), -1)
            base_x, base_y = int(pts[1][0]), int(max(10, pts[1][1]-30))
            cv2.putText(frame, f"Yaw:{yaw:.1f} MARo:{speech_metrics['mar_outer']:.2f} MARi:{speech_metrics['mar_inner']:.2f}",
                        (base_x, base_y), FONT, 0.42, (255,255,255), 1)
            # Show thresholds (debug)
            cv2.putText(frame, f"Thr(o/i/d):{speech_metrics['thr_out']:.2f}/{speech_metrics['thr_in']:.2f}/{speech_metrics['thr_dist']:.1f}",
                        (base_x, base_y-18), FONT, 0.38, (200,200,200), 1)

            if speaking_persist:
                cv2.putText(frame, f"SPEAKING (conf:{speech_confidence:.2f})",
                            (base_x, base_y-36), FONT, 0.6, (0,200,0), 2)
            if gaze_side:
                cv2.putText(frame, "LOOKING AWAY", (base_x, base_y-54), FONT, 0.6, (0,165,255), 2)
            if head_move_flag:
                cv2.putText(frame, "HEAD MOVING", (base_x, base_y-72), FONT, 0.6, (0,165,255), 2)
            if hand_cover:
                cv2.putText(frame, "HAND COVERING", (base_x, base_y-90), FONT, 0.6, (0,0,255), 2)

    # Combine frame-level flags
    any_speaking = any(ff["speaking"] for ff in face_flags)
    any_head_move = any(ff["head_move"] for ff in face_flags)
    any_gaze_away = any(ff["gaze_side"] for ff in face_flags)
    any_hand_cover = any(ff.get("hand_covering", False) for ff in face_flags)

    reasons = []
    max_speech_conf = 0.0
    best_speech_metrics = {"mar_outer":0.0, "mar_inner":0.0, "lip_distance":0.0}

    if multi_person_flag: reasons.append("MULTIPLE PEOPLE")
    if empty_chair_flag: reasons.append("EMPTY CHAIR")
    if phone_flag: reasons.append("PHONE DETECTED")
    if any_speaking:
        for ff in face_flags:
            if ff["speaking"] and ff["speech_confidence"] > max_speech_conf:
                max_speech_conf = ff["speech_confidence"]
                best_speech_metrics = ff["speech_metrics"]
        reasons.append(f"SPEAKING (conf:{max_speech_conf:.2f})")
    if any_head_move: reasons.append("HEAD MOVEMENT")
    if any_gaze_away: reasons.append("LOOKING AWAY")
    if any_hand_cover: reasons.append("HAND COVERING")

    # Header
    status_color = (0,255,0) if len(reasons)==0 else (0,0,255)
    txt = "OK" if len(reasons)==0 else "; ".join(reasons)
    cv2.rectangle(frame, (0,0), (frame.shape[1], 30), (0,0,0), -1)
    cv2.putText(frame, f"STATUS: {txt}", (10,20), FONT, 0.6, status_color, 2)

    # Counts
    cv2.putText(frame, f"Persons: {len(detected_persons)} Phones: {len(detected_phones)} Chairs: {len(detected_chairs)}",
                (10, frame.shape[0]-10), FONT, 0.5, (200,200,200), 1)

    # Logging cooldown
    now_ts = time.time()
    if len(reasons) > 0 and (now_ts - last_log_time > LOG_COOLDOWN_SEC):
        log_event("; ".join(reasons), frame.copy(),
                  persons_count=len(detected_persons),
                  phones_count=len(detected_phones),
                  chairs_count=len(detected_chairs),
                  mar_outer=best_speech_metrics["mar_outer"],
                  mar_inner=best_speech_metrics["mar_inner"],
                  lip_distance=best_speech_metrics["lip_distance"],
                  speech_confidence=max_speech_conf)
        last_log_time = now_ts

    cv2.imshow("Enhanced Proctoring", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
hands.close()
print("[INFO] Enhanced proctoring system closed.")
print(f"[INFO] Logs: {LOG_DIR}")
print(f"[INFO] Dashboard: {DASHBOARD_FILE}")
