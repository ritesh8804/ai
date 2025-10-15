# app_proctoring_tight.py
"""
Tighter AI Proctoring Streamlit app.
- Temporal smoothing (sustained events required)
- Debounce/cooldown for logs
- Stricter thresholds for person / verification
- Minimal logs & evidence capture only for sustained violations
"""
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime, timedelta
import os, time
from collections import deque

# --------------------
# CONFIG (tune these)
# --------------------
st.set_page_config(page_title="AI Proctoring (Tight)", layout="wide")
SAVE_PATH = "proctoring_evidence"
os.makedirs(SAVE_PATH, exist_ok=True)

YOLO_WEIGHTS = "yolov8n.pt"
AUTHORIZED_FACE = "student_reference.jpg"
VERIFY_INTERVAL = 8.0  # seconds between verify attempts during test
YOLO_PERSON_CONF = 0.55  # require stronger person confidence
SUSTAIN_FRAMES_REQUIRED = 8  # number of frames (approx) for sustained event
MOTION_THRESHOLD = 10.0  # motion score threshold
HEAD_DEPTH_MIN = 20.0  # ignore tiny depth noise
HEAD_DEPTH_MAX = 500.0
EAR_BLINK_THRESHOLD = 0.16
SPEAK_LIP_DIST = 12
VIOLATION_THRESHOLD = 2  # how many sustained violations to block
LOG_COOLDOWN_SEC = 20  # don't log same warning repeatedly within cooldown
FRAME_BUFFER_SIZE = 90
VIDEO_FPS = 15
VIDEO_SIZE = (640, 480)

# --------------------
# MODELS / RESOURCES
# --------------------
@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

yolo = load_yolo_model(YOLO_WEIGHTS)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --------------------
# SESSION STATE
# --------------------
if "phase" not in st.session_state:
    st.session_state.phase = "verify"  # verify -> test
if "verified" not in st.session_state:
    st.session_state.verified = False
if "verify_conf" not in st.session_state:
    st.session_state.verify_conf = None
if "last_verify_time" not in st.session_state:
    st.session_state.last_verify_time = 0.0

if "frame_buf" not in st.session_state:
    st.session_state.frame_buf = deque(maxlen=FRAME_BUFFER_SIZE)
if "logs" not in st.session_state:
    st.session_state.logs = deque(maxlen=100)
if "violation_count" not in st.session_state:
    st.session_state.violation_count = 0
if "blocked" not in st.session_state:
    st.session_state.blocked = False

# temporal counters for smoothing
if "counters" not in st.session_state:
    st.session_state.counters = {
        "no_person": 0,
        "multi_person": 0,
        "face_not_visible": 0,
        "excess_motion": 0,
        "head_depth": 0,
        "speaking": 0,
        "spoof": 0,
        "unauthorized": 0,
    }
if "last_log_times" not in st.session_state:
    st.session_state.last_log_times = {}

# last frame store for motion
if "last_frame_gray" not in st.session_state:
    st.session_state.last_frame_gray = None

# --------------------
# UTILITIES
# --------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def push_log(kind, detail=""):
    key = kind + "|" + detail.split("|")[0] if detail else kind
    last = st.session_state.last_log_times.get(key)
    now = time.time()
    if last and now - last < LOG_COOLDOWN_SEC:
        return  # suppress due to cooldown
    entry = f"{now_str()} | {kind}{(' | ' + detail) if detail else ''}"
    st.session_state.logs.append(entry)
    st.session_state.last_log_times[key] = now
    print("[LOG]", entry)

def save_image(frame, tag):
    ts = now_str()
    fname = f"{tag}_{ts}.jpg"
    path = os.path.join(SAVE_PATH, fname)
    cv2.imwrite(path, frame)
    push_log("EvidenceSaved", fname)
    return path

def save_video(frames, tag):
    ts = now_str()
    fname = f"{tag}_{ts}.mp4"
    path = os.path.join(SAVE_PATH, fname)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, VIDEO_FPS, VIDEO_SIZE)
    for f in frames:
        out.write(cv2.resize(f, VIDEO_SIZE))
    out.release()
    push_log("EvidenceSaved", fname)
    return path

def get_landmark(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)

def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def ear(landmarks, left_idx, right_idx, w, h):
    def ratio(indices):
        p1 = get_landmark(landmarks, indices[0], w, h)
        p2 = get_landmark(landmarks, indices[1], w, h)
        p3 = get_landmark(landmarks, indices[2], w, h)
        p4 = get_landmark(landmarks, indices[3], w, h)
        p5 = get_landmark(landmarks, indices[4], w, h)
        p6 = get_landmark(landmarks, indices[5], w, h)
        return (euclid(p2, p6) + euclid(p3, p5)) / (2.0 * euclid(p1, p4))
    try:
        return (ratio(left_idx) + ratio(right_idx)) / 2.0
    except Exception:
        return 1.0

# --------------------
# VERIFICATION PAGE (strict)
# --------------------
def verification_page():
    st.header("Identity Verification (Required)")
    st.write("Take a clear selfie. Verification must pass before the exam starts.")

    c1, c2 = st.columns([2, 1])
    with c1:
        cam = st.camera_input("Take verification photo")
    with c2:
        st.write("Registered reference:")
        if os.path.exists(AUTHORIZED_FACE):
            st.image(AUTHORIZED_FACE, use_column_width=True)
        else:
            st.warning("Reference image not found. Place student_reference.jpg in working folder.")

    if cam:
        file_bytes = np.asarray(bytearray(cam.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured")

        try:
            t0 = time.time()
            res = DeepFace.verify(frame, AUTHORIZED_FACE, enforce_detection=True, model_name="Facenet512")
            dur = time.time() - t0
            verified = bool(res.get("verified", False))
            # pick any numeric distance if present - smaller distance often means better match
            conf = None
            for k in ("distance", "cosine", "score"):
                if k in res:
                    conf = res[k]; break
            st.session_state.verify_conf = conf
            if verified:
                push_log("Verified", f"time={dur:.2f}s conf={conf}")
                st.success(f"Verified (time {dur:.2f}s) — proceeding to test.")
                st.session_state.verified = True
                st.session_state.phase = "test"
            else:
                push_log("VerificationFailed", f"conf={conf}")
                st.error("Verification failed. Please retake photo and ensure face is visible and not occluded.")
        except Exception as e:
            push_log("VerificationError", str(e))
            st.error("Verification error: " + str(e))

# --------------------
# PROCESS FRAME (tighter rules)
# --------------------
def process_frame(frame_bgr):
    warnings = []
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # YOLO detection with stricter person confidence
    try:
        res = yolo(frame_bgr, verbose=False)
        boxes = res[0].boxes
        person_count = 0
        for box in boxes:
            # depending on ultralytics version, fields differ
            xyxy = box.xyxy[0].tolist()
            cls = int(box.cls[0]) if hasattr(box, "cls") else (int(box[5]) if len(box) > 5 else -1)
            conf = float(box.conf[0]) if hasattr(box, "conf") else (float(box[4]) if len(box) > 4 else 0.0)
            name = res[0].names[cls] if cls >= 0 and cls < len(res[0].names) else str(cls)
            if name == "person" and conf >= YOLO_PERSON_CONF:
                person_count += 1
                x1, y1, x2, y2 = map(int, xyxy[:4])
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame_bgr, f"person {conf:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1)
    except Exception as e:
        print("YOLO err:", e)
        person_count = 0

    # Face mesh
    person_face_present = False
    try:
        fm = face_mesh.process(rgb)
        if fm.multi_face_landmarks:
            person_face_present = True
    except Exception:
        person_face_present = False

    # update counters with smoothing
    cs = st.session_state.counters
    # no person vs multi-person
    if person_count == 0:
        cs["no_person"] += 1
        cs["multi_person"] = 0
    elif person_count > 1:
        cs["multi_person"] += 1
        cs["no_person"] = 0
    else:
        cs["no_person"] = 0
        cs["multi_person"] = 0

    if not person_face_present and person_count == 1:
        cs["face_not_visible"] += 1
    else:
        cs["face_not_visible"] = 0

    # landmark-based checks (only if face present)
    liveness_flags = {"blink": False, "motion": False, "depth": False}
    if person_face_present:
        lm = fm.multi_face_landmarks[0].landmark
        # EAR blink
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        ear_val = ear(lm, left_eye, right_eye, w, h)
        if ear_val < EAR_BLINK_THRESHOLD:
            # count as single blink event but not a violation
            liveness_flags["blink"] = True

        # motion score (compare with previous grayscale frame)
        cur_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if st.session_state.last_frame_gray is not None:
            diff = cv2.absdiff(st.session_state.last_frame_gray, cur_gray)
            motion_score = np.mean(diff)
            if motion_score > MOTION_THRESHOLD:
                cs["excess_motion"] += 1
            else:
                cs["excess_motion"] = max(0, cs["excess_motion"] - 1)
        st.session_state.last_frame_gray = cur_gray

        # head pose (solvePnP) - depth check
        try:
            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ])
            image_points = np.array([
                get_landmark(lm, 1, w, h),
                get_landmark(lm, 152, w, h),
                get_landmark(lm, 263, w, h),
                get_landmark(lm, 33, w, h),
                get_landmark(lm, 287, w, h),
                get_landmark(lm, 57, w, h)
            ], dtype="double")
            focal = w
            cm = np.array([[focal, 0, w/2], [0, focal, h/2], [0,0,1]])
            dist = np.zeros((4,1))
            ok, rvec, tvec = cv2.solvePnP(model_points, image_points, cm, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                depth = float(tvec[2][0])
                # if depth outside reasonable head distance (very large or very small) count up
                if depth < HEAD_DEPTH_MIN or depth > HEAD_DEPTH_MAX:
                    cs["head_depth"] += 1
                else:
                    cs["head_depth"] = max(0, cs["head_depth"] - 1)
        except Exception:
            pass

        # lip distance -> speaking detection (sustained)
        try:
            top = get_landmark(lm, 14, w, h)
            bottom = get_landmark(lm, 13, w, h)
            lip_dist = abs(top[1] - bottom[1])
            if lip_dist > SPEAK_LIP_DIST:
                cs["speaking"] += 1
            else:
                cs["speaking"] = max(0, cs["speaking"] - 1)
        except Exception:
            pass

    # periodic verification (reduce frequency)
    cur_t = time.time()
    if cur_t - st.session_state.last_verify_time > VERIFY_INTERVAL:
        st.session_state.last_verify_time = cur_t
        try:
            vr = DeepFace.verify(frame_bgr, AUTHORIZED_FACE, enforce_detection=False, model_name="Facenet512")
            if not vr.get("verified", False):
                cs["unauthorized"] += 1
            else:
                cs["unauthorized"] = max(0, cs["unauthorized"] - 1)
        except Exception:
            # keep previous state if verify fails
            pass

    # check sustained thresholds and only then produce official warnings
    sustained_warnings = []

    if cs["no_person"] >= SUSTAIN_FRAMES_REQUIRED:
        sustained_warnings.append("No person detected (sustained)")
    if cs["multi_person"] >= SUSTAIN_FRAMES_REQUIRED:
        sustained_warnings.append("Multiple persons detected (sustained)")
    if cs["face_not_visible"] >= SUSTAIN_FRAMES_REQUIRED:
        sustained_warnings.append("Face not visible (sustained)")
    if cs["excess_motion"] >= SUSTAIN_FRAMES_REQUIRED:
        sustained_warnings.append("Excessive head motion (sustained)")
    if cs["head_depth"] >= SUSTAIN_FRAMES_REQUIRED:
        sustained_warnings.append("Large head pose / depth change (sustained)")
    if cs["speaking"] >= SUSTAIN_FRAMES_REQUIRED:
        sustained_warnings.append("Speaking (sustained)")
    if cs["unauthorized"] >= 2:  # require fewer cycles for identity mismatch
        sustained_warnings.append("Unauthorized person (sustained)")

    # treat 'spoof' if liveness flags are very poor (rare)
    live_count = 0
    if person_face_present:
        # consider blink or motion or depth as live cues
        if liveness_flags["blink"]:
            live_count += 1
        if cs["excess_motion"] > 0:
            live_count += 1
        if cs["head_depth"] > 0:
            live_count += 1
    if person_face_present and live_count < 2:
        cs["spoof"] += 1
    else:
        cs["spoof"] = max(0, cs["spoof"] - 1)
    if cs["spoof"] >= SUSTAIN_FRAMES_REQUIRED:
        sustained_warnings.append("Low liveness / possible spoof (sustained)")

    # if sustained warnings occur: log once, increment violation_count, capture minimal evidence
    if sustained_warnings:
        reason = "; ".join(sustained_warnings)
        push_log("SustainedViolation", reason)
        st.session_state.violation_count += 1
        # only save minimal evidence on sustained event
        if len(st.session_state.frame_buf) > 0:
            # save latest image and short clip
            img_path = save_image(st.session_state.frame_buf[-1], "violation")
            vid_path = save_video(list(st.session_state.frame_buf), "violation_clip")
            push_log("SavedEvidencePaths", f"{img_path} | {vid_path}")

    # draw light overlays for admin clarity
    cv2.putText(frame_bgr, f"Violations: {st.session_state.violation_count}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if st.session_state.violation_count else (0,200,0), 2)

    # append frame to buffer for potential evidence
    st.session_state.frame_buf.append(frame_bgr.copy())

    # block if threshold reached
    if st.session_state.violation_count >= VIOLATION_THRESHOLD and not st.session_state.blocked:
        st.session_state.blocked = True
        push_log("BLOCKED", f"violation_count={st.session_state.violation_count}")
        # save immediate evidence
        if len(st.session_state.frame_buf) > 0:
            save_image(st.session_state.frame_buf[-1], "blocked_evidence")
            save_video(list(st.session_state.frame_buf), "blocked_clip")

    return frame_bgr, sustained_warnings

# --------------------
# UI PAGES
# --------------------
def render_verify():
    st.title("Exam — Identity Verification")
    st.write("Clear selfie required. This must match the registered student.")
    col1, col2 = st.columns([2,1])
    with col1:
        cam = st.camera_input("Take verification photo")
    with col2:
        if os.path.exists(AUTHORIZED_FACE):
            st.image(AUTHORIZED_FACE, caption="Registered reference")
        else:
            st.warning("Reference missing: place student_reference.jpg")

    if cam:
        data = np.asarray(bytearray(cam.read()), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured")
        try:
            res = DeepFace.verify(frame, AUTHORIZED_FACE, enforce_detection=True, model_name="Facenet512")
            ok = res.get("verified", False)
            conf = None
            for k in ("distance","cosine","score"):
                if k in res:
                    conf = res[k]; break
            st.session_state.verify_conf = conf
            if ok:
                push_log("Verified", f"conf={conf}")
                st.success("Identity verified — starting test.")
                st.session_state.verified = True
                st.session_state.phase = "test"
            else:
                push_log("VerifyFailed", f"conf={conf}")
                st.error("Verification failed. Retake photo.")
        except Exception as e:
            push_log("VerifyError", str(e))
            st.error("Verification error: " + str(e))

def render_test():
    st.title("Exam In Progress")
    if st.session_state.blocked:
        st.error("Test blocked due to repeated violations. Contact admin.")
    col_student, col_admin = st.columns([1.0, 1.0])

    with col_student:
        st.subheader("Student view")
        st.write("Keep camera visible. Only minimal feedback is shown.")
        cam = st.camera_input("Camera (capture a frame frequently)")
        if cam:
            data = np.asarray(bytearray(cam.read()), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            # process and display
            processed, warns = process_frame(frame)
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_column_width=True)
            if warns:
                st.warning("Attention detected — proctor notified.")
            else:
                st.success("OK")
        else:
            st.info("Awaiting student camera input...")

    with col_admin:
        st.subheader("Admin view")
        if st.session_state.frame_buf:
            st.image(cv2.cvtColor(st.session_state.frame_buf[-1], cv2.COLOR_BGR2RGB), use_column_width=True)
        else:
            st.info("No frames yet.")
        st.write("Recent logs:")
        for l in list(reversed(list(st.session_state.logs)))[0:20]:
            st.text(l)
        st.metric("Violations", st.session_state.violation_count)
        if st.button("Force save last evidence"):
            if st.session_state.frame_buf:
                p = save_image(st.session_state.frame_buf[-1], "manual_save")
                st.success(f"Saved: {p}")
            else:
                st.info("No frames to save.")
        if st.button("Reset session (admin)"):
            st.session_state.phase = "verify"
            st.session_state.verified = False
            st.session_state.frame_buf.clear()
            st.session_state.logs.clear()
            st.session_state.violation_count = 0
            st.session_state.blocked = False
            st.session_state.last_frame_gray = None
            st.session_state.counters = {k:0 for k in st.session_state.counters}
            push_log("Reset", "admin reset")
            st.experimental_rerun()

# --------------------
# APP ROUTING
# --------------------
def main():
    st.sidebar.title("Proctor Controls")
    st.sidebar.write(f"Phase: {st.session_state.phase}")
    if st.sidebar.button("Go to Verify"):
        st.session_state.phase = "verify"
    if st.sidebar.button("Go to Test (force)"):
        st.session_state.phase = "test"
    if st.sidebar.button("Show evidence folder"):
        st.write("Evidence saved to:", os.path.abspath(SAVE_PATH))

    if st.session_state.phase == "verify":
        render_verify()
    else:
        if not st.session_state.verified:
            st.warning("Student not verified. Verification required.")
            if st.button("Admin override to start test"):
                push_log("AdminOverride", "Test started without verification")
                st.session_state.phase = "test"
        render_test()

if __name__ == "__main__":
    main()
