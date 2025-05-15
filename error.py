import cv2
import numpy as np
import face_recognition

image_path = "C:\\Users\\nandw\\Downloads\\face1_converted.jpg"
image_bgr = cv2.imread(image_path)

if image_bgr is None:
    print("Image failed to load.")
    exit()

print("Original dtype:", image_bgr.dtype)
print("Original shape:", image_bgr.shape)
print("Pixel value range: min =", image_bgr.min(), "max =", image_bgr.max())

# Force conversion to uint8 and 3 channels
if image_bgr.dtype != np.uint8 or image_bgr.shape[2] != 3:
    print("Forcing conversion to 8-bit RGB...")
    image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Now test face detection
try:
    face_locations = face_recognition.face_locations(image_rgb)
    print(f"Face detection successful: {len(face_locations)} face(s) found.")
except Exception as e:
    print("Still failed:", e)
