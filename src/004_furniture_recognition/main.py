from common.computer_vision import webcam_context, frames
from common.cv_drawing import draw_box
from common.utils import get_device
from furniture_recognition import get_model


def main():
    device = get_device()
    model = get_model()
    model.to(device)

    with webcam_context() as camera:
        for frame in frames(camera):
            results = model(frame, stream=True)

            for r in results:
                for box in r.boxes:
                    draw_box(
                        frame,
                        box,
                        color=(0, 255, 0),
                        thickness=2,
                        draw_label=True,
                        label_map=model.names,
                        draw_confidence=True,
                    )


if __name__ == "__main__":
    main()
