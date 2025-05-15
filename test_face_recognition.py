import face_recognition
import cv2
import numpy as np

# Load the image using OpenCV directly
image_path = "C:\\Users\\nandw\\Downloads\\RGB.jpg"
image_bgr = cv2.imread(image_path)

if image_bgr is None:
    print("Failed to load image. Check the file path and name.")
    exit()

# Convert from BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
print("Image loaded and converted to RGB successfully.")

# Try to detect face locations
try:
    face_locations = face_recognition.face_locations(image_rgb)
    print(f"Found {len(face_locations)} face(s) in the image.")
except Exception as e:
    print("Face detection failed:", e)
    exit()

# Draw rectangles on the faces
for top, right, bottom, left in face_locations:
    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

# Show the image
cv2.imshow("Detected Faces", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
