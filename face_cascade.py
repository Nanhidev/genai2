import cv2
import os

output_dir = "detected_faces_cascade"
os.makedirs(output_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

video_path = "Videos/1.webm"  # Change this to "video.mp4" if using a video file
cap = cv2.VideoCapture(video_path)

frame_count = 0  # Frame counter to avoid overwriting saved faces

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when the video ends

    frame_count += 1  # Increment frame counter

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i, (x, y, w, h) in enumerate(faces):
        face = frame[y:y+h, x:x+w]

        face_filename = os.path.join(output_dir, f"face_{frame_count}_{i}.jpg")
        cv2.imwrite(face_filename, face)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()





