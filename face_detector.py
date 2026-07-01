# ==========================================
# Face Detection with Photo Capture
# Author: Donald (with Anna 😄)
# ==========================================

# Import required libraries
import cv2
import os
from datetime import datetime


# Load the pre-trained Haar Cascade classifier
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Check that the classifier loaded successfully
if face_detector.empty():
    print(" Failed to load the face detection model.")
    exit()


# Open the default webcam

camera = cv2.VideoCapture(0)

# Check that the webcam opened successfully
if not camera.isOpened():
    print(" Failed to access the webcam.")
    exit()


# Create the captures folder if it doesn't exist
CAPTURE_FOLDER = "captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

print("✅ Face Detection Started")
print("Press 'S' to save a photo.")
print("Press 'Q' to quit.")


# Main Loop

while True:

    # Capture one frame
    success, frame = camera.read()

    if not success:
        print("❌ Failed to read frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Display number of detected faces
    cv2.putText(
        frame,
        f"Faces Detected: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Display instructions
    cv2.putText(
        frame,
        "Press S to Save | Press Q to Quit",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Show the webcam feed
    cv2.imshow("OpenCV Face Detection", frame)

    # Read keyboard input
    key = cv2.waitKey(1) & 0xFF

    # ------------------------------------------
    # Save a photo when 'S' is pressed
    # ------------------------------------------
    if key == ord("s"):

        # Generate a unique filename using the current date and time
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.jpg")

        # Build the full file path
        filepath = os.path.join(CAPTURE_FOLDER, filename)

        # Save the current frame
        cv2.imwrite(filepath, frame)

        print(f" Photo saved successfully: {filepath}")

    # ------------------------------------------
    # Quit when 'Q' is pressed
    # ------------------------------------------
    elif key == ord("q"):
        break

# ------------------------------------------
# Release resources
# ------------------------------------------
camera.release()
cv2.destroyAllWindows()