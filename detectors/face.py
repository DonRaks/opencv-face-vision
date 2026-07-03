"""
face.py

contains the FaceDetector class that uses OpenCV's Haar Cascades to detect faces and eyes in a given frame.
"""

import cv2
from utils.drawing import Drawing


class FaceDetector:
    """
    Detects faces and eyes using OpenCV Haar Cascades.
    """

    def __init__(self):

        # Load face detection model
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Load eye detection model
        self.eye_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        if self.face_detector.empty():
            raise RuntimeError("Failed to load face detector.")

        if self.eye_detector.empty():
            raise RuntimeError("Failed to load eye detector.")

    def detect(self, frame):
        """
        Detect faces and eyes.

        Args:
            frame: Camera frame

        Returns:
            frame: Frame with rectangles drawn
            face_count: Number of detected faces
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:

            # Draw face rectangle
            Drawing.draw_face(frame, x, y, w, h)

            # Region of Interest
            face_gray = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]

            # Detect eyes
            eyes = self.eye_detector.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
            )

            # Region of Interest
            face_gray = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]

            # Detect eyes
            eyes = self.eye_detector.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )

            # Draw eye rectangles
            for (ex, ey, ew, eh) in eyes:

                Drawing.draw_eye(face_color, ex, ey, ew, eh)

        return frame, len(faces)