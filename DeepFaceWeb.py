import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime
import os
import time

# ========================
# INIT
# ========================
st.set_page_config(page_title="AI Proctoring System", layout="wide")

yolo_model = YOLO("yolov8n.pt")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

SAVE_PATH = "proctoring_evidence"
os.makedirs(SAVE_PATH, exist_ok=True)

AUTHORIZED_FACE = "student_reference.jpg"  # reference image

if "logs" not in st.session_state:
    st.session_state.logs = []
if "frame" not in st.session_state:
    st.session_state.frame = None
if "violation_count" not in st.session_state:
    st.session_state.violation_count = 0
if "blocked" not in st.session_state:
    st.session_state.blocked = False
if "last_verify_time" not in st.session_state:
    st.session_state.last_verify_time = 0
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "blink_count" not in st.session_state:
    st.session_state.blink_count = 0


# ========================
# UTILITIES
# ========================
def log_event(event, detail=""):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    entry = f"{timestamp} | {event} | {detail}"
    st.session_state.logs.append(entry)
    print(entry)
    if st.session_state.frame is not None:
        cv2.imwrite(os.path.join(SAVE_PATH, f"{event}_{timestamp}.jpg"), st.session_state.frame)


def get_landmark_point(landmarks, index, w, h):
    return int(landmarks[index].x * w), int(landmarks[index].y * h)


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, left_indices, right_indices, w, h):
    def ratio(indices):
        p1 = get_landmark_point(landmarks, indices[0], w, h)
        p2 = get_landmark_point(landmarks, indices[1], w, h)
        p3 = get_landmark_point(landmarks, indices[2], w, h)
        p4 = get_landmark_point(landmarks, indices[3], w, h)
        p5 = get_landmark_point(landmarks, indices[4], w, h)
        p6 = get_landmark_point(landmarks, indices[5], w, h)
        return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4))

    return (ratio(left_indices) + ratio(right_indices)) / 2.0


# ========================
# DETECTION PIPELINE
# ========================
def detect(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model(frame)
    detections = results[0].boxes.data
    person_count = sum(
        1 for det in detections if results[0].names[int(det[5])] == "person" and det[4] > 0.5
    )

    result = face_mesh.process(rgb)
    person_present = result.multi_face_landmarks is not None
    warnings = []
    liveness_flags = {"blink": False, "motion": False, "depth": False}

    # ---- Basic Presence ----
    if person_count == 0:
        warnings.append("No person detected")
    elif person_count > 1:
        warnings.append("Multiple persons detected")

    if not person_present and person_count == 1:
        warnings.append("Empty chair / Face not visible")

    # ---- DeepFace Emotion Detection ----
    try:
        analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        emotion = analysis[0]["dominant_emotion"]
        cv2.putText(frame, f"Emotion: {emotion}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if emotion in ["angry", "fear", "disgust", "surprise"]:
            warnings.append(f"Unusual Emotion: {emotion}")
    except Exception as e:
        print("Emotion detection error:", e)

    # ---- DeepFace Verification (every 5 sec) ----
    current_time = time.time()
    if current_time - st.session_state.last_verify_time > 5:
        st.session_state.last_verify_time = current_time
        try:
            verify_result = DeepFace.verify(frame, AUTHORIZED_FACE,
                                            enforce_detection=False, model_name="Facenet512")
            if not verify_result["verified"]:
                warnings.append("Unauthorized person detected")
                cv2.putText(frame, "Identity: Unauthorized", (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Identity: Verified", (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            print("Verification error:", e)

    # ---- MediaPipe Tracking ----
    if person_present:
        landmarks = result.multi_face_landmarks[0].landmark

        # === Blink Detection (Liveness) ===
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        ear = eye_aspect_ratio(landmarks, left_eye, right_eye, w, h)
        if ear < 0.18:
            st.session_state.blink_count += 1
            liveness_flags["blink"] = True
        cv2.putText(frame, f"Blink Count: {st.session_state.blink_count}", (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

        # === Motion Check ===
        if st.session_state.last_frame is not None:
            diff = cv2.absdiff(cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            motion_score = np.mean(diff)
            if motion_score > 5:
                liveness_flags["motion"] = True
        st.session_state.last_frame = frame.copy()

        # === Head Pose (Depth change) ===
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        image_points = np.array([
            get_landmark_point(landmarks, 1, w, h),
            get_landmark_point(landmarks, 152, w, h),
            get_landmark_point(landmarks, 263, w, h),
            get_landmark_point(landmarks, 33, w, h),
            get_landmark_point(landmarks, 287, w, h),
            get_landmark_point(landmarks, 57, w, h),
        ], dtype="double")

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            depth = translation_vec[2][0]
            if abs(depth) > 400:
                liveness_flags["depth"] = True

        # === Lip Movement ===
        top_lip = get_landmark_point(landmarks, 14, w, h)
        bottom_lip = get_landmark_point(landmarks, 13, w, h)
        lip_distance = abs(top_lip[1] - bottom_lip[1])
        speaking = lip_distance > 10
        cv2.putText(frame, f"Speaking: {'Yes' if speaking else 'No'}", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if not speaking else (0, 0, 255), 2)
        if speaking:
            warnings.append("Speaking Detected")

    # ---- Liveness Decision ----
    live_score = sum(liveness_flags.values())
    if live_score < 2:
        warnings.append("⚠️ Spoof Detected (Low Liveness Score)")
        log_event("Warning", "Possible Spoof Detected")
    else:
        cv2.putText(frame, "Liveness: Real", (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---- Flag Handling ----
    for warn in warnings:
        log_event("Warning", warn)
        st.session_state.violation_count += 1

    if st.session_state.violation_count >= 3:
        st.session_state.blocked = True
        log_event("BLOCKED", "Test terminated due to repeated violations")

    return frame, warnings


# ========================
# UI
# ========================
view = st.sidebar.radio("Select View", ["🎓 Student View", "🕵️ Admin View"])

if view == "🎓 Student View":
    st.header("Student Exam Window")
    if st.session_state.blocked:
        st.error("🚫 Test Blocked due to multiple violations.")
    else:
        camera = st.camera_input("Enable Camera")
        if camera:
            frame_bytes = camera.getvalue()
            np_frame = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
            st.session_state.frame = frame
            processed, warnings = detect(frame)
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")

            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.success("All good!")

elif view == "🕵️ Admin View":
    st.header("Admin Proctoring Dashboard")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("### Live Feed")
        if st.session_state.frame is not None:
            st.image(cv2.cvtColor(st.session_state.frame, cv2.COLOR_BGR2RGB), channels="RGB")
        else:
            st.info("Waiting for student feed...")

    with col2:
        st.write("### Logs")
        for log in reversed(st.session_state.logs[-10:]):
            st.text(log)
        st.metric("Total Violations", st.session_state.violation_count)
        if st.session_state.blocked:
            st.error("Test Blocked ❌")
