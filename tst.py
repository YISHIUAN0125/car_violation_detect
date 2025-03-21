import cv2
from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt

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

# Function to check if a car is violating parking rules
def check_parking_violation(x1, y1, x2, y2, red_mask):
    # Define areas around the car to check for red lines
    above_area = red_mask[max(0, y1):y1-10, x1:x2]
    below_area = red_mask[y2:y2+10, x1:x2]

    # Check if red lines exist in these regions
    above_red = np.any(above_area > 0)
    below_red = np.any(below_area > 0)

    return (above_red and below_red)

model = YOLO(r'runs\detect\train\weights\best_ncnn_model')
image = cv2.imread(r'0816_record\20250320_113840.jpg')
results = model.predict(image)
boxes = results[0].boxes
for box in boxes:
    id = int(box.cls)
    xy_arr = box.xyxy.cpu()
    coordi = np.array(xy_arr)
    x1, y1, x2, y2 = coordi[0]
    
print(x1, y1, x2, y2)
red_mask = detect_red_lines(image)
print(check_parking_violation(int(x1), int(y1), int(x2), int(y2), red_mask))
plt.subplot(1, 3, 1)
plt.title('check parking violation')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.title('Red Mask')
plt.imshow(red_mask, cmap = 'gray')
plt.subplot(1, 3, 3)
plt.title('YOLO Detection')
plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
plt.show()

