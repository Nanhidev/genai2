from mtcnn import MTCNN
import cv2

detector = MTCNN()
img = cv2.imread("OPENCV/person.jpeg")
faces = detector.detect_faces(img)

for face in faces:
    x, y, w, h = face["box"]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()