from contextlib import contextmanager
from typing import Generator

import cv2
from pyglet.window.key import ESCAPE


@contextmanager
def webcam_context() -> Generator[cv2.VideoCapture, None]:
    video_capture = cv2.VideoCapture(0)
    try:
        if not video_capture.isOpened():
            raise RuntimeError("Could not open video device.")
        yield video_capture
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


def frames(video_capture: cv2.VideoCapture) -> Generator[cv2.Mat, None]:
    should_exit = False
    while not should_exit:
        ret, frame = video_capture.read()
        try:
            if not ret:
                raise RuntimeError("Could not read frame from video device.")
            yield frame
        finally:
            cv2.imshow('Output', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == ESCAPE & 0xFF:
                should_exit = True


def get_face_cascade(casc_path: str) -> cv2.CascadeClassifier:
    try:
        face_cascade = cv2.CascadeClassifier(casc_path)
        if face_cascade.empty():
            raise IOError(f"Could not load cascade file at {casc_path}")
    except AttributeError:
        raise RuntimeError(
            "Please make sure OpenCV is installed correctly.",
            "You may need to manually download 'haarcascade_frontalface_default.xml' and place it in your script's directory."
        )
    return face_cascade
