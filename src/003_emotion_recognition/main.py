import sys

import cv2
import torch
from torchvision import transforms

from common.utils import get_device
from emotions import get_trained_emotions_model, EMOTIONS

if __name__ == "__main__":
    model = get_trained_emotions_model()

    try:
        cascPath = "./models/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        if faceCascade.empty():
            raise IOError(f"Could not load cascade file at {cascPath}")
    except AttributeError:
        print("Please make sure OpenCV is installed correctly.", file=sys.stderr)
        print(
            "You may need to manually download 'haarcascade_frontalface_default.xml' and place it in your script's directory.",
            file=sys.stderr
        )
        sys.exit(1)
    except IOError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    data_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array (OpenCV) to PIL Image
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((48, 48)),  # Resize to 48x48
        transforms.ToTensor(),  # Convert to a PyTorch tensor
        # Normalize with the mean and std dev used during training
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    video_capture = cv2.VideoCapture(0)

    device = get_device()

    if not video_capture.isOpened():
        print("Error: Could not open video device.", file=sys.stderr)
        sys.exit(1)

    print("Webcam opened. Press 'q' in the video window to quit.")

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Could not read frame from video device.", file=sys.stderr)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_roi = gray[y:y + h, x:x + w]

            try:
                face_tensor = data_transform(face_roi).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Skipping frame due to transform error: {e}")
                continue

            with torch.no_grad():
                outputs = model(face_tensor)
                predicted_index = torch.argmax(outputs, dim=1).item()
                emotion = EMOTIONS[predicted_index]

            cv2.putText(
                frame,
                emotion,
                (x, y - 10),  # Position the text just above the box
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Font scale
                (0, 255, 0),  # Font color (Green)
                2  # Font thickness
            )

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
