"""
main.py

Entry point for the Face Vision application.
"""

import cv2

from services.camera import Camera
from detectors.face import FaceDetector
from utils.drawing import Drawing


def main():
    """
    Main application loop.
    """

    # Initialize the camera
    camera = Camera()

    # Initialize the face detector
    detector = FaceDetector()

    print("Face Detection Started")
    print("Press 'Q' to quit.")

    while True:

        # Read a frame from the webcam
        success, frame = camera.read()

        if not success:
            print("Failed to read frame.")
            break

        # Detect faces and eyes
        frame, face_count = detector.detect(frame)

        # Display the number of detected faces
        

        Drawing.draw_text(frame, f"Faces Detected: {face_count}", (10, 30))

        # Display the video
        cv2.imshow("OpenCV Face Vision", frame)

        # Read keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()