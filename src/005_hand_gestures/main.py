import cv2
from ultralytics.engine.results import Results

from common.computer_vision import webcam_context, frames
from common.cv_drawing import draw_box, draw_keypoints
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
            detections: list[Results] = model(frame, stream=True)

            for r in detections:
                for hand_index, (box, keypoints) in enumerate(zip(r.boxes, r.keypoints)):
                    draw_box(frame, box, label_map={0: f"Hand {hand_index + 1}"})

                    draw_keypoints(
                        frame,
                        keypoints,
                        draw_indices=True,
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
