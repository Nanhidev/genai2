from deepface import DeepFace
import cv2
import numpy as np

# Load the image and detect the face
image_path = "OPENCV/person.jpeg"
detected_face = DeepFace.detectFace(image_path, detector_backend="retinaface")

# Convert from float (0-1) to uint8 (0-255)
detected_face = (detected_face * 255).astype(np.uint8)

# Convert RGB to BGR (DeepFace returns RGB, but OpenCV expects BGR)
detected_face = cv2.cvtColor(detected_face, cv2.COLOR_RGB2BGR)





# Display the detected face
cv2.imshow("Detected Face", detected_face)
cv2.waitKey(0)
cv2.destroyAllWindows()
