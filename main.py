import cv2

from services.camera import Camera


camera = Camera()

while True:

    success, frame = camera.read()

    if not success:
        break

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()