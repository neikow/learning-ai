from ultralytics import YOLO

from common.utils import get_device

_model: YOLO | None = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolo11n-pose.pt")
    return _model


def train_model(epochs: int = 10):
    model = get_model()
    model.train(
        data="hand-keypoints.yaml",
        epochs=epochs,
        imgsz=640,
        device=get_device()
    )
