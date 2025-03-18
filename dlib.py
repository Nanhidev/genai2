import dlib
import cv2

# Load Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Open a video file or webcam (0 for webcam, or provide a video file path)
video_path = "https://www.youtube.com/watch?v=e6iNJgMvDi4"  
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends or there's an issue

    # Convert frame to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray, 1)  # Change `1` to `2` or `3` for more accuracy

    # Draw rectangles around detected faces
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the video frame with detections
    cv2.imshow("Face Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
