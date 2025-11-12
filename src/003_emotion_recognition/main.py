import cv2
import torch
from torchvision import transforms

from common.computer_vision import webcam_context, frame_context, get_face_cascade
from common.utils import get_device
from emotions import get_trained_emotions_model, EMOTIONS

if __name__ == "__main__":
    model = get_trained_emotions_model()

    casc_path = "./models/haarcascade_frontalface_default.xml"
    face_cascade = get_face_cascade(casc_path)

    data_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array (OpenCV) to PIL Image
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((48, 48)),  # Resize to 48x48
        transforms.ToTensor(),  # Convert to a PyTorch tensor
        # Normalize with the mean and std dev used during training
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    device = get_device()

    with webcam_context() as video_capture:
        while True:
            with frame_context(video_capture) as frame:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(
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
