"""
camera.py

Handles all webcam operations.
"""

import cv2


class Camera:
    """
    A simple wrapper around OpenCV's VideoCapture.
    """

    def __init__(self, camera_index: int = 0):
        """
        Initialize the webcam.

        Args:
            camera_index (int): Webcam index (default is 0).
        """
        self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Failed to access the webcam.")

    def read(self):
        """
        Read a frame from the webcam.

        Returns:
            tuple:
                success (bool)
                frame (numpy.ndarray)
        """
        return self.camera.read()

    def release(self):
        """
        Release the webcam.
        """
        self.camera.release()