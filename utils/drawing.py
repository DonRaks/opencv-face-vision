"""
drawing.py

Contains helper methods for drawing objects on frames.
"""

import cv2


class Drawing:
    """Utility class for drawing on images."""

    @staticmethod
    def draw_face(frame, x, y, w, h):
        """
        Draw a green rectangle around a detected face.
        """
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    @staticmethod
    def draw_eye(face_region, x, y, w, h):
        """
        Draw a blue rectangle around a detected eye.
        """
        cv2.rectangle(
            face_region,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    @staticmethod
    def draw_text(frame, text, position):
        """
        Draw text on the frame.
        """
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )