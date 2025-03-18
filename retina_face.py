from retinaface import RetinaFace
import cv2 

img=cv2.imread("OPENCV/person.jpeg")
faces = RetinaFace.detect_faces("OPENCV/person.jpeg")

for key in faces.keys():
    identity = faces[key]
    x, y, w, h = identity["facial_area"]
    cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()