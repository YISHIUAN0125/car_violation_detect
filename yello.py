import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 黃網線偵測函式
def detect_yellow_net(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 轉換為 HSV 色彩空間
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_mask = cv2.erode(yellow_mask, (3,3), iterations=2)
    yellow_mask = cv2.dilate(yellow_mask, (5,5), iterations=10)
    return yellow_mask

# 檢查車輛是否違規（任三邊有黃網線即算違規）
def check_parking_violation(x1, y1, x2, y2, yellow_mask):
    # 檢查車輛四周是否有黃線
    above_area = yellow_mask[max(0, y1-20):y1, x1:x2]
    below_area = yellow_mask[y2:y2+20, x1:x2]
    left_area = yellow_mask[y1:y2, max(0, x1-20):x1]
    right_area = yellow_mask[y1:y2, x2:x2+20]

    above_yellow = np.any(above_area > 0)
    below_yellow = np.any(below_area > 0)
    left_yellow = np.any(left_area > 0)
    right_yellow = np.any(right_area > 0)

    # 計算有幾邊有黃線
    yellow_count = sum([above_yellow, below_yellow, left_yellow, right_yellow])
    print(yellow_count)
    return yellow_count >= 3



# 加載 YOLO 車輛偵測模型
model = YOLO(r'runs\detect\train\weights\best_ncnn_model')
# cv2.VideoCapture()
try:
    car_bbox = []
    image_path = r"0816_record\Screenshot 2025-03-18 121818.png"
    image = cv2.imread(image_path)
    yellow_mask = detect_yellow_net(image)
    result = model.predict(image)
    boxes = result[0].boxes

    for box in boxes:
        id = int(box.cls)
        xy_arr = box.xyxy.cpu().numpy()
        x1, y1, x2, y2 = map(int, xy_arr[0])
        if id == 0:
            car_bbox.append((x1, y1, x2, y2))

    for i, (x1, y1, x2, y2) in enumerate(car_bbox):
        is_violating = check_parking_violation(x1, y1, x2, y2, yellow_mask)
        if is_violating:
            print(f"Car {i+1} is violating parking rule by being on yellow net line.")
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f'yellow grid violation', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            print(f"Car {i+1} is not violating parking rule.")

    annotated_img = result[0].plot()
    cv2.imwrite("result.jpg", image)
    plt.figure(figsize=(12, 6))
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("YOLO detection result")
    
    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("violating car detection result")
    
    plt.subplot(1,3,3)
    plt.imshow(yellow_mask, cmap="gray")
    plt.axis("off")
    plt.title("yellow net line detection result")
    
    plt.show()
except Exception as e:
    print(e)
    exit(1)
