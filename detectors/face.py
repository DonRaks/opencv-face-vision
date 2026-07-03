# ==========================================
# Face Detection with Eye Detection
# Author: Donald (with Anna)
# ==========================================

# Import required libraries
import cv2
import os
from datetime import datetime

# ------------------------------------------
# Load the face detector
# ------------------------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Failed to load the face detection model.")
    exit()


# Load the eye detector

eye_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if eye_detector.empty():
    print("Failed to load the eye detection model.")
    exit()


# Open the webcam

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Failed to access the webcam.")
    exit()


# Create the captures folder

CAPTURE_FOLDER = "captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

print("Face Detection Started")
print("Press 'S' to save a photo.")
print("Press 'Q' to quit.")

# ------------------------------------------
# Main loop
# ------------------------------------------
while True:

    # Read a frame from the webcam
    success, frame = camera.read()

    if not success:
        print("Failed to read frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Loop through every detected face
    for (x, y, w, h) in faces:

        # Draw a green rectangle around the face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Create Region of Interest (ROI)
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # Detect eyes inside the face
        eyes = eye_detector.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        # Draw blue rectangles around each detected eye
        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                face_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
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

    # Display controls
    cv2.putText(
        frame,
        "Press S to Save | Press Q to Quit",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Show the frame
    cv2.imshow("OpenCV Face Detection", frame)

    # Read keyboard input
    key = cv2.waitKey(1) & 0xFF

    # Save a photo
    if key == ord("s"):

        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.jpg")
        filepath = os.path.join(CAPTURE_FOLDER, filename)

        cv2.imwrite(filepath, frame)

        print(f"Photo saved: {filepath}")

    # Quit the application
    elif key == ord("q"):
        break

# ------------------------------------------
# Clean up
# ------------------------------------------
camera.release()
cv2.destroyAllWindows()