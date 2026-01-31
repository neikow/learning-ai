import cv2.typing
from numpy import ndarray
from ultralytics.engine.results import Boxes, Keypoints


def draw_box(
        frame: ndarray,
        box: Boxes,
        *,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3,
        draw_label: bool = True,
        label_map: dict[int, str] | None = None,
        draw_confidence: bool = True,
) -> None:
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label_text = ""
    if draw_label:
        cls = int(box.cls[0])
        if label_map and cls in label_map:
            label_text += label_map[cls]
        else:
            label_text += str(cls)

    if draw_confidence:
        confidence = round(float(box.conf[0]) * 100) / 100
        if label_text:
            label_text += " | "
        label_text += f"{confidence:.2f}"

    if label_text:
        org = (x1, y1 - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        text_color = color
        text_thickness = 2

        cv2.putText(
            frame,
            label_text,
            org,
            font,
            fontScale,
            text_color,
            text_thickness
        )


def draw_keypoints(
        frame: ndarray,
        keypoints: Keypoints,
        *,
        point_color: tuple[int, int, int] = (0, 255, 0),
        text_color: tuple[int, int, int] = (0, 255, 0),
        radius: int = 5,
        thickness: int = -1,
        draw_indices: bool = True,
) -> None:
    for index, pt in enumerate(keypoints.xy[0]):
        x, y = pt[:2]
        cv2.circle(frame, (x, y), radius, point_color, thickness)

        if draw_indices:
            cv2.putText(
                frame,
                str(index),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1
            )
