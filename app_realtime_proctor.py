# app_realtime_proctor.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from deepface import DeepFace
from collections import deque, defaultdict
import os, time, threading
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Realtime AI Proctoring", layout="wide")
SAVE_PATH = "proctoring_evidence"
os.makedirs(SAVE_PATH, exist_ok=True)

YOLO_WEIGHTS = "yolov8n.pt"              # change if needed
AUTHORIZED_FACE = "student_reference.jpg"
VERIFY_INTERVAL = 6.0                    # seconds between identity verify checks while streaming
FRAME_BUFFER_SIZE = 150                  # number of frames saved in buffer (~10 sec at 15fps)
VIDEO_FPS = 15
VIOLATION_THRESHOLD = 2                  # number of sustained violations (occurrences) before block
SUSTAIN_FRAMES_REQUIRED = 10             # how many frames of evidence required to consider sustained
YOLO_PERSON_CONF = 0.55                  # stricter person conf
MOTION_THRESHOLD = 9.0
EAR_THRESHOLD = 0.16
LIP_DIST_SPEAK = 12
LOG_COOLDOWN = 20.0                      # seconds to suppress duplicate logs
# ----------------------------------------

# ---------------- helpers ----------------
def now_ts():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_image(frame, tag, extra=""):
    ts = now_ts()
    name = f"{tag}_{ts}{('_'+extra) if extra else ''}.jpg"
    path = os.path.join(SAVE_PATH, name)
    cv2.imwrite(path, frame)
    return path

def save_video(frames, tag):
    ts = now_ts()
    fname = f"{tag}_{ts}.mp4"
    path = os.path.join(SAVE_PATH, fname)
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    return path

def get_landmark(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)

def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def ear(landmarks, lef, rig, w, h):
    def r(indices):
        p1 = get_landmark(landmarks, indices[0], w, h)
        p2 = get_landmark(landmarks, indices[1], w, h)
        p3 = get_landmark(landmarks, indices[2], w, h)
        p4 = get_landmark(landmarks, indices[3], w, h)
        p5 = get_landmark(landmarks, indices[4], w, h)
        p6 = get_landmark(landmarks, indices[5], w, h)
        return (euclid(p2, p6) + euclid(p3, p5)) / (2.0 * euclid(p1, p4))
    try:
        return (r(lef) + r(rig)) / 2.0
    except Exception:
        return 1.0

# ---------------- Streamlit session state ----------------
if "phase" not in st.session_state:
    st.session_state.phase = "verify"   # "verify" -> "stream"
if "verified" not in st.session_state:
    st.session_state.verified = False
if "violation_count" not in st.session_state:
    st.session_state.violation_count = 0
if "blocked" not in st.session_state:
    st.session_state.blocked = False
if "logs" not in st.session_state:
    st.session_state.logs = deque(maxlen=200)
if "last_log_time" not in st.session_state:
    st.session_state.last_log_time = defaultdict(lambda: 0.0)

def push_log(kind, detail=""):
    key = kind + "|" + (detail.split("|")[0] if detail else "")
    now = time.time()
    if now - st.session_state.last_log_time[key] < LOG_COOLDOWN:
        return
    st.session_state.last_log_time[key] = now
    entry = f"{now_ts()} | {kind}{(' | ' + detail) if detail else ''}"
    st.session_state.logs.append(entry)
    print(entry)

# ---------------- The real-time transformer ----------------
mp_face_mesh = mp.solutions.face_mesh

class ProctorTransformer(VideoTransformerBase):
    """
    This runs in worker thread and handles real-time frame processing.
    It maintains its own buffer and counters and writes evidence files when necessary.
    """
    def __init__(self):
        # lazy-loaded models to avoid heavy imports at module import time
        self.yolo = None
        self.face_mesh = None
        self.last_verify_time = 0.0
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        # counters for smoothing
        self.counters = defaultdict(int)
        self.lock = threading.Lock()
        self.last_frame_gray = None
        self.blocked = False
        self.verified_local = st.session_state.verified  # snapshot; can be refreshed by UI
        # small cooldown to avoid saving too frequently
        self.last_evidence_saved = 0.0

    def lazy_init(self):
        if self.yolo is None:
            try:
                self.yolo = YOLO(YOLO_WEIGHTS)
                print("YOLO loaded in transformer")
            except Exception as e:
                print("Error loading YOLO:", e)
                self.yolo = None
        if self.face_mesh is None:
            self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                   refine_landmarks=True, min_detection_confidence=0.5,
                                                   min_tracking_confidence=0.5)

    def process(self, frame):
        """
        frame: av.VideoFrame-like; use frame.to_ndarray(format='bgr24') to get BGR numpy array.
        """
        self.lazy_init()
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        bgr = img.copy()
        now = time.time()

        # draw verification status
        cv2.putText(bgr, f"Verified: {st.session_state.verified}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if st.session_state.verified else (0,0,255), 2)

        # short-circuit if blocked
        if st.session_state.blocked:
            cv2.putText(bgr, "TEST BLOCKED", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            # still show frames to admin, but no checks
            self.frame_buffer.append(bgr.copy())
            return bgr

        # YOLO detections (try/except because model may fail)
        person_count = 0
        try:
            if self.yolo:
                yres = self.yolo(bgr, verbose=False)
                boxes = yres[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                    cls = int(box.cls[0]) if hasattr(box, "cls") else -1
                    name = yres[0].names[cls] if 0 <= cls < len(yres[0].names) else str(cls)
                    x1,y1,x2,y2 = map(int, xyxy[:4])
                    color = (0,200,0) if name=="person" else (200,200,0)
                    cv2.rectangle(bgr, (x1,y1),(x2,y2), color, 2)
                    cv2.putText(bgr, f"{name} {conf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    if name == "person" and conf >= YOLO_PERSON_CONF:
                        person_count += 1
        except Exception as e:
            print("YOLO in transformer error:", e)

        # face mesh
        face_present = False
        fm = None
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            fm = self.face_mesh.process(rgb)
            if fm and fm.multi_face_landmarks:
                face_present = True
        except Exception as e:
            pass

        # update person counters with smoothing
        if person_count == 0:
            self.counters["no_person"] += 1
            self.counters["multi_person"] = 0
        elif person_count > 1:
            self.counters["multi_person"] += 1
            self.counters["no_person"] = 0
        else:
            self.counters["no_person"] = 0
            self.counters["multi_person"] = 0

        if not face_present and person_count == 1:
            self.counters["face_not_visible"] += 1
        else:
            self.counters["face_not_visible"] = 0

        # landmark-based checks if face present
        live_flags = {"blink": False, "motion": False, "depth": False}
        if face_present:
            lm = fm.multi_face_landmarks[0].landmark
            # EAR blink
            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]
            ear_val = ear(lm, left_eye, right_eye, w, h)
            cv2.putText(bgr, f"EAR:{ear_val:.2f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,200),1)
            if ear_val < EAR_THRESHOLD:
                live_flags["blink"] = True

            # grayscale motion
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if self.last_frame_gray is not None:
                diff = cv2.absdiff(self.last_frame_gray, gray)
                mscore = np.mean(diff)
                cv2.putText(bgr, f"Motion:{mscore:.1f}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,255),1)
                if mscore > MOTION_THRESHOLD:
                    self.counters["excess_motion"] += 1
                else:
                    self.counters["excess_motion"] = max(0, self.counters["excess_motion"] - 1)
            self.last_frame_gray = gray

            # depth / head pose
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
                    get_landmark(lm, 57, w, h),
                ], dtype="double")
                focal_length = w
                center = (w/2, h/2)
                camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0,0,1]])
                dist_coeffs = np.zeros((4,1))
                ok, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    depth = float(tvec[2][0])
                    cv2.putText(bgr, f"Depth:{depth:.1f}", (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,200),1)
                    # simple heuristic: if depth extreme, mark counter
                    if depth < 10 or depth > 600:
                        self.counters["head_depth"] += 1
                    else:
                        self.counters["head_depth"] = max(0, self.counters["head_depth"] - 1)
            except Exception:
                pass

            # speaking (lip dist)
            try:
                top = get_landmark(lm, 14, w, h)
                bottom = get_landmark(lm, 13, w, h)
                lip = abs(top[1] - bottom[1])
                cv2.putText(bgr, f"Lip:{lip}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,200),1)
                if lip > LIP_DIST_SPEAK:
                    self.counters["speaking"] += 1
                else:
                    self.counters["speaking"] = max(0, self.counters["speaking"] - 1)
            except Exception:
                pass

        # periodic DeepFace verification (not on every frame!)
        if now - self.last_verify_time > VERIFY_INTERVAL:
            self.last_verify_time = now
            if os.path.exists(AUTHORIZED_FACE) and self.yolo is not None:
                try:
                    # try to verify; permit failure gracefully
                    result = DeepFace.verify(bgr, AUTHORIZED_FACE, enforce_detection=False, model_name="Facenet512")
                    verified = bool(result.get("verified", False))
                    if not verified:
                        self.counters["unauthorized"] += 1
                    else:
                        self.counters["unauthorized"] = max(0, self.counters["unauthorized"] - 1)
                    # record small text
                    if "distance" in result:
                        cv2.putText(bgr, f"Dist:{result['distance']:.2f}", (10,175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0),1)
                except Exception as e:
                    # ignore heavy exceptions
                    print("DeepFace verify error:", e)

        # check sustained conditions
        sustained = []
        if self.counters["no_person"] >= SUSTAIN_FRAMES_REQUIRED:
            sustained.append("No person (sustained)")
        if self.counters["multi_person"] >= SUSTAIN_FRAMES_REQUIRED:
            sustained.append("Multiple persons (sustained)")
        if self.counters["face_not_visible"] >= SUSTAIN_FRAMES_REQUIRED:
            sustained.append("Face not visible (sustained)")
        if self.counters["excess_motion"] >= SUSTAIN_FRAMES_REQUIRED:
            sustained.append("Excess motion (sustained)")
        if self.counters["head_depth"] >= SUSTAIN_FRAMES_REQUIRED:
            sustained.append("Head pose / depth (sustained)")
        if self.counters["speaking"] >= SUSTAIN_FRAMES_REQUIRED:
            sustained.append("Speaking (sustained)")
        if self.counters["unauthorized"] >= 2:
            sustained.append("Unauthorized (sustained)")

        # if sustained violations => record evidence and increment violation count
        if sustained:
            reason = "; ".join(sustained)
            # push log (via streamlit session state visible to UI)
            push_log("SustainedViolation", reason)
            st.session_state.violation_count += 1
            # only save evidence at most once every LOG_COOLDOWN seconds
            if now - self.last_evidence_saved > LOG_COOLDOWN:
                self.last_evidence_saved = now
                # save latest image
                try:
                    saved_img = save_image(self.frame_buffer[-1] if len(self.frame_buffer)>0 else bgr, "violation", extra=str(st.session_state.violation_count))
                except Exception:
                    saved_img = save_image(bgr, "violation", extra=str(st.session_state.violation_count))
                # save short clip from buffer
                try:
                    clip = list(self.frame_buffer)[-min(len(self.frame_buffer), 90):]
                    vid = save_video(clip, "violation_clip")
                except Exception as e:
                    print("Video save error:", e)
                    vid = None
                push_log("EvidenceSaved", f"{saved_img} | {vid}")

        # update block if too many sustained violations
        if st.session_state.violation_count >= VIOLATION_THRESHOLD:
            st.session_state.blocked = True
            push_log("BLOCKED", f"violations={st.session_state.violation_count}")
            # capture final evidence
            try:
                save_image(bgr, "blocked_evidence")
                save_video(list(self.frame_buffer), "blocked_clip")
            except Exception:
                pass

        # add to frame buffer
        self.frame_buffer.append(bgr.copy())

        # overlay minimal admin info
        cv2.putText(bgr, f"Violations: {st.session_state.violation_count}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,0,255) if st.session_state.violation_count>0 else (0,200,0), 2)

        return bgr

# ---------------- UI layout ----------------
st.title("Realtime AI Proctoring — Live Stream")

# left: verification / controls; right: live viewer + admin
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Setup & Controls")
    st.write("1) Verification (required). 2) Start live proctoring stream. 3) Admin panel shows logs and evidence.")
    # Verification step: accept single photo to verify before going to stream
    st.subheader("Identity verification")
    if os.path.exists(AUTHORIZED_FACE):
        st.image(AUTHORIZED_FACE, caption="Registered reference", use_column_width=True)
    else:
        st.warning("Reference image student_reference.jpg not found in working directory. Place it there.")
    photo = st.camera_input("Take verification selfie (required)")
    if photo is not None:
        data = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Captured", use_column_width=True)
        # try verify (blocking; single call)
        try:
            with st.spinner("Verifying..."):
                res = DeepFace.verify(img, AUTHORIZED_FACE, enforce_detection=True, model_name="Facenet512")
            verified = bool(res.get("verified", False))
            st.session_state.verified = verified
            if verified:
                push_log("Verified", f"verification ok")
                st.success("Identity verified — you may start the live proctoring stream.")
                st.session_state.phase = "stream"
            else:
                push_log("VerifyFailed", f"{res}")
                st.error("Verification FAILED. Ensure face matches the reference and retake.")
        except Exception as e:
            push_log("VerifyError", str(e))
            st.error("Verification error: " + str(e))

    st.markdown("---")
    st.subheader("Manual controls")
    if st.button("Reset session (admin)"):
        st.session_state.verified = False
        st.session_state.phase = "verify"
        st.session_state.violation_count = 0
        st.session_state.blocked = False
        st.session_state.logs.clear()
        push_log("Reset", "admin reset")
        st.experimental_rerun()

    if st.button("Force clear violations"):
        st.session_state.violation_count = 0
        st.session_state.blocked = False
        push_log("ClearedViolations", "admin cleared")

    st.write("Evidence path:", os.path.abspath(SAVE_PATH))
    st.write("Recent logs:")
    for l in list(reversed(list(st.session_state.logs)))[0:30]:
        st.text(l)

with col2:
    st.header("Live stream (Student view & Admin overlay)")
    if not st.session_state.verified:
        st.info("Please verify identity first on the left panel.")
    # start the webrtc streamer with our transformer if verified or admin override
    if st.session_state.verified or st.button("Start stream (admin override)"):
        webrtc_ctx = webrtc_streamer(
            key="proctor",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_transformer_factory=ProctorTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
            desired_playing_latency=0.5,
        )

    # Admin panel: show last saved evidence files from the folder
    st.markdown("---")
    st.subheader("Admin Panel")
    st.metric("Violations", st.session_state.violation_count)
    if st.session_state.blocked:
        st.error("Test is BLOCKED due to repeated violations.")
    st.write("Recent logs:")
    for l in list(reversed(list(st.session_state.logs)))[0:20]:
        st.text(l)

    # show evidence list (last 10 files)
    files = sorted(os.listdir(SAVE_PATH), reverse=True)
    if files:
        st.write("Recent evidence files:")
        for f in files[:10]:
            st.write(f)
            if f.lower().endswith((".jpg", ".png")):
                st.image(os.path.join(SAVE_PATH, f), width=240)
            else:
                st.write(f"Video: {os.path.join(SAVE_PATH, f)}")
    else:
        st.info("No evidence saved yet.")

st.markdown("---")
st.caption("Notes: This demo runs models locally in the page; performance depends on CPU/GPU and browser. For production use, consider a TURN server for WebRTC and offloading heavy verification to a GPU backend.")
