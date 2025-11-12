import math

import cv2

from common.computer_vision import webcam_context, frames
from furniture_recognition import get_model


def main():
    model = get_model()

    with webcam_context() as camera:
        for frame in frames(camera):
            results = model(frame, stream=True)

            names = model.names

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])

                    org = [x1, y1 - 10]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (0, 255, 0)
                    thickness = 2

                    cv2.putText(frame, f"{names[cls]} | {round(confidence * 100)}%", org, font, fontScale, color,
                                thickness)


if __name__ == "__main__":
    main()
