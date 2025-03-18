from deepface import DeepFace
import cv2
import numpy as np

# Open a video file or webcam (0 for webcam, or provide a video file path)
video_path = "https://www.youtube.com/watch?v=e6iNJgMvDi4" 
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends or there's an issue

    try:
        # Detect face (DeepFace expects an image array, not a file path)
        detected_face = DeepFace.detectFace(frame, detector_backend="retinaface")

        # Convert from float (0-1) to uint8 (0-255)
        detected_face = (detected_face * 255).astype(np.uint8)

        # Convert RGB to BGR (for OpenCV display)
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_RGB2BGR)

        # Show the detected face
        cv2.imshow("Detected Face", detected_face)

    except Exception as e:
        print("No face detected:", e)  # Handle cases where no face is found

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
