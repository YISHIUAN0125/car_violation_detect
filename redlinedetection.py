import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the uploaded image
image_path = r"0816_record\redline\img0816_165720.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying with matplotlib

# Define the bounding box of the detected car
x1, y1 = int(170.27), int(40.209)
x2, y2 = int(491.4), int(178.28)

# Remove the car area from the image
image_no_car = image.copy()
image_no_car[y1:y2, x1:x2] = 0

# Function to detect red lines (red color mask)
def detect_red_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks to detect red color in two ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    return red_mask

# Detect red lines
red_mask = detect_red_lines(image_no_car)

# Define areas around the car to check for red lines
above_area = red_mask[max(0, y1 - 50):y1, x1:x2]
below_area = red_mask[y2:y2 + 50, x1:x2]
left_area = red_mask[y1:y2, max(0, x1 - 50):x1]
right_area = red_mask[y1:y2, x2:x2 + 50]

# Visualize the areas
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(above_area, cmap="gray")
plt.title("Above Area")
plt.subplot(2, 2, 2)
plt.imshow(below_area, cmap="gray")
plt.title("Below Area")
plt.subplot(2, 2, 3)
plt.imshow(left_area, cmap="gray")
plt.title("Left Area")
plt.subplot(2, 2, 4)
plt.imshow(right_area, cmap="gray")
plt.title("Right Area")
plt.show()

# Check if red lines exist in these regions
above_red = np.any(above_area > 0)
below_red = np.any(below_area > 0)
left_red = np.any(left_area > 0)
right_red = np.any(right_area > 0)
print(f"Above Red: {above_red}")
print(f"Below Red: {below_red}")
print(f"Left Red: {left_red}")
print(f"Right Red: {right_red}")

if (above_red and below_red) or (left_red and right_red):
    print("Car is violating parking rule by being on red line.")
else:
    print("Car is not violating parking rule.")

# Visualize the result
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.subplot(1, 3, 2)
plt.imshow(red_mask, cmap="gray")
plt.title("Red Line Mask")
plt.subplot(1, 3, 3)
cv2.rectangle(image_rgb, (x1, max(0, y1 - 50)), (x2, y1), (255, 0, 0), 2)  # Above region
cv2.rectangle(image_rgb, (x1, y2), (x2, y2 + 50), (0, 0, 255), 2)  # Below region
cv2.rectangle(image_rgb, (max(0, x1 - 50), y1), (x1, y2), (0, 255, 0), 2)  # Left region
cv2.rectangle(image_rgb, (x2, y1), (x2 + 50, y2), (255, 255, 0), 2)  # Right region
plt.imshow(image_rgb)
plt.title("Car Detection with Bounding Box")
plt.show()




