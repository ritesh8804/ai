import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime
import time
import os

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

if "logs" not in st.session_state:
    st.session_state.logs = []
if "frame" not in st.session_state:
    st.session_state.frame = None
if "violation_count" not in st.session_state:
    st.session_state.violation_count = 0
if "blocked" not in st.session_state:
    st.session_state.blocked = False

# ========================
# UTILITIES
# ========================
def log_event(event, detail=""):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    entry = f"{timestamp} | {event} | {detail}"
    st.session_state.logs.append(entry)
    print(entry)
    # Save snapshot
    if st.session_state.frame is not None:
        cv2.imwrite(os.path.join(SAVE_PATH, f"{event}_{timestamp}.jpg"), st.session_state.frame)


def get_landmark_point(landmarks, index, w, h):
    return int(landmarks[index].x * w), int(landmarks[index].y * h)


def detect(frame):
    """Run YOLO + FaceMesh full analysis."""
    h, w, _ = frame.shape
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
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # FaceMesh detection
    result = face_mesh.process(rgb)
    person_present = result.multi_face_landmarks is not None

    warnings = []

    # ---- Empty Chair or Multiple People ----
    if person_count == 0:
        warnings.append("No person detected")
    elif person_count > 1:
        warnings.append("Multiple persons detected")

    if not person_present and person_count == 1:
        warnings.append("Empty chair / Face not visible")

    # ---- Advanced tracking ----
    if person_present:
        landmarks = result.multi_face_landmarks[0].landmark

        # === Eye Tracking ===
        left_outer = get_landmark_point(landmarks, 33, w, h)
        left_inner = get_landmark_point(landmarks, 133, w, h)
        left_iris = get_landmark_point(landmarks, 468, w, h)

        right_outer = get_landmark_point(landmarks, 362, w, h)
        right_inner = get_landmark_point(landmarks, 263, w, h)
        right_iris = get_landmark_point(landmarks, 473, w, h)

        def eye_direction(iris, inner, outer):
            center_x = (inner[0] + outer[0]) // 2
            if iris[0] < center_x - 5:
                return "Left"
            elif iris[0] > center_x + 5:
                return "Right"
            else:
                return "Center"

        left_dir = eye_direction(left_iris, left_inner, left_outer)
        right_dir = eye_direction(right_iris, right_inner, right_outer)
        cv2.putText(frame, f"Left Eye: {left_dir}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Right Eye: {right_dir}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if left_dir != "Center" or right_dir != "Center":
            warnings.append(f"Looking Away ({left_dir}/{right_dir})")

        # === Lips Tracking ===
        top_lip = get_landmark_point(landmarks, 14, w, h)
        bottom_lip = get_landmark_point(landmarks, 13, w, h)
        lip_distance = abs(top_lip[1] - bottom_lip[1])
        speaking = lip_distance > 10
        cv2.putText(frame, f"Speaking: {'Yes' if speaking else 'No'}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if not speaking else (0, 0, 255), 2)
        if speaking:
            warnings.append("Speaking Detected")

        # === Head Pose Estimation ===
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

        nose_tip = image_points[0]
        nose_end_point_3d = np.array([[0, 0, 1000.0]])
        nose_end_point_2d, _ = cv2.projectPoints(
            nose_end_point_3d, rotation_vec, translation_vec, camera_matrix, dist_coeffs
        )

        p1 = (int(nose_tip[0]), int(nose_tip[1]))
        p2 = (int(nose_end_point_2d[0][0][0]), int(nose_end_point_2d[0][0][1]))
        cv2.line(frame, p1, p2, (255, 0, 255), 3)

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

        cv2.putText(frame, f"Head: {direction}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        if direction != "Straight":
            warnings.append(f"Head Pose: {direction}")

    # === Flag Handling ===
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
