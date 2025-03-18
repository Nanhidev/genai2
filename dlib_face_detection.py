import dlib
import cv2

detector = dlib.get_frontal_face_detector()
img = cv2.imread("OPENCV/person.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convet to img dlib
# gray = cv2.equalizeHist(gray)  # Improves contrast
faces = detector(gray, 1)  # Change `1` to `2` or `3` if needed

for face in faces:
    print(face)
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
