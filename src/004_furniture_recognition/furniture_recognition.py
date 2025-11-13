from ultralytics import YOLO

from common.utils import get_device

_model: YOLO | None = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolo11n.pt")
    return _model


def train_model(epochs: int = 100):
    model = get_model()
    model.train(
        data="furniture_data.yaml",
        epochs=epochs,
        imgsz=640,
        name="furniture_recognition_model",
        device=get_device()
    )
