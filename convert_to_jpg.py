from PIL import Image

# Open image and convert to 8-bit RGB
img = Image.open("C:\\Users\\nandw\\Downloads\\face1.png")  # Use actual file name and extension
img = img.convert("RGB")
img.save("C:\\Users\\nandw\\Downloads\\face1_converted.jpg", format="JPEG", quality=95)
print("Saved clean RGB 8-bit JPEG as face1_converted.jpg")
