# ==========================================
# Face Detection using OpenCV
# Author: Donald (with Anna )
# ==========================================

# Import OpenCV
import cv2

# ------------------------------------------
# Load the pre-trained Haar Cascade classifier
# ------------------------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Check that the classifier loaded successfully
if face_detector.empty():
    print("Failed to load the face detection model.")
    exit()

# ------------------------------------------
# Open the default webcam
# ------------------------------------------
camera = cv2.VideoCapture(0)

# Check that the webcam opened successfully
if not camera.isOpened():
    print("Could not access the webcam.")
    exit()

print("✅ Face Detection Started")
print("Press 'q' to quit.")

# ------------------------------------------
# Main loop
# ------------------------------------------
while True:

    # Capture one frame from the webcam
    success, frame = camera.read()

    # Stop if we couldn't read a frame
    if not success:
        print(" Failed to read frame.")
        break

    # Convert the frame to grayscale
    # Haar Cascades work on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,              # Image to draw on
            (x, y),             # Top-left corner
            (x + w, y + h),     # Bottom-right corner
            (0, 255, 0),        # Green color (BGR)
            2                   # Thickness
        )

    # Display the number of faces detected
    cv2.putText(
        frame,
        f"Faces Detected: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Display the instructions
    cv2.putText(
        frame,
        "Press Q to Quit",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show the webcam feed
    cv2.imshow("OpenCV Face Detection", frame)

    # Wait for keyboard input
    key = cv2.waitKey(1) & 0xFF

    # Quit if 'q' is pressed
    if key == ord("q"):
        break

# ------------------------------------------
# Clean up
# ------------------------------------------
camera.release()
cv2.destroyAllWindows()