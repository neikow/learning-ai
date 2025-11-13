import cv2

from common.computer_vision import webcam_context, frames
from common.utils import get_device
from hand_gestures import get_model

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # Palm
    (5, 6), (6, 7), (7, 8),  # Index finger
    (9, 10), (10, 11), (11, 12),  # Middle finger
    (13, 14), (14, 15), (15, 16),  # Ring finger
    (17, 18), (18, 19), (19, 20)  # Little finger
]


def main():
    device = get_device()
    model = get_model()
    model.to(device)

    with webcam_context() as camera:
        for frame in frames(camera):
            results = model(frame, stream=True)

            names = model.names

            for hand_index, r in enumerate(results):
                for box, keypoints in zip(r.boxes, r.keypoints.xy):
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    for index, (x, y) in enumerate(keypoints):
                        x, y = int(x), int(y)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        cv2.putText(
                            frame,
                            str(index),
                            (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1
                        )

                    for connection in CONNECTIONS:
                        start_idx, end_idx = connection
                        x_start, y_start = keypoints[start_idx]
                        x_end, y_end = keypoints[end_idx]
                        cv2.line(
                            frame,
                            (int(x_start), int(y_start)),
                            (int(x_end), int(y_end)),
                            (0, 255, 0),
                            2
                        )


if __name__ == "__main__":
    # train_model()
    main()
