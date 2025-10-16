from deepface import DeepFace
import cv2

# Paths
REFERENCE_IMAGE = "reference.jpg"   # Your stored reference image
TEST_IMAGE = "capture.jpg"          # Image to verify (you can capture from webcam)

# Optional: Capture from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
if ret:
    cv2.imwrite(TEST_IMAGE, frame)

# Verify face
try:
    result = DeepFace.verify(TEST_IMAGE, REFERENCE_IMAGE, enforce_detection=True)
    if result["verified"]:
        print("✅ Identity Verified")
    else:
        print("❌ Identity Not Verified")
    print("Similarity / Distance:", result["distance"])
except Exception as e:
    print("Error:", e)
