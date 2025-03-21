import cv2
import numpy as np
from ultralytics import YOLO

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

    return (above_red or below_red)

if __name__ == '__main__':
    num = 0
    model = YOLO(r'runs\detect\train\weights\best_ncnn_model')
    car_bbox = []
    frame = cv2.imread(r'0816_record\20250320_113840.jpg')
    red_mask = detect_red_lines(frame)
    result = model.predict(frame)
    boxes = result[0].boxes
    for box in boxes:
        id = int(box.cls)
        xy_arr = box.xyxy.cpu()
        coordi = np.array(xy_arr)
        x1, y1, x2, y2 = coordi[0]
        if id == 0:
            car_bbox.append((int(x1), int(y1), int(x2), int(y2)))
    for i, (x1, y1, x2, y2) in enumerate(car_bbox):
        is_violating = check_parking_violation(x1, y1, x2, y2, red_mask)
        if is_violating:
            print(f"Car {i+1} is violating parking rule by being on red line.")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'red line violation', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imwrite(f'./0816_record/redline/car.jpg', frame)
        else:
            print(f"Car {i+1} is not violating parking rule.")
    cv2.imshow('frame', result[0].plot())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # cap = cv2.VideoCapture('video0207_1541.mp4')
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 取得影像寬度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')          # Update codec to 'mp4v'
    # out = cv2.VideoWriter(r'argument_vid.mp4', fourcc, 30.0, (width,  height))  # 產生空的影片
    # ret, frame = cap.read()
    # try:
    #     while ret:
    #         car_bbox = []
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         red_mask = detect_red_lines(frame)
    #         result = model.predict(frame)
    #         boxes = result[0].boxes
    #         for box in boxes:
    #             id = int(box.cls)
    #             xy_arr = box.xyxy.cpu()
    #             coordi = np.array(xy_arr)
    #             x1, y1, x2, y2 = coordi[0]
    #             if id == 0:
    #                 car_bbox.append((int(x1), int(y1), int(x2), int(y2)))
    #         for i, (x1, y1, x2, y2) in enumerate(car_bbox):
    #             is_violating = check_parking_violation(x1, y1, x2, y2, red_mask)
    #             if is_violating:
    #                 print(f"Car {i+1} is violating parking rule by being on red line.")
    #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #                 cv2.putText(frame, f'Car {i+1} - Violation', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    #                 cv2.imwrite(f'./0816_record/redline/car_{num}.jpg', frame)
    #             else:
    #                 print(f"Car {i+1} is not violating parking rule.")
    #         annotated_img = result[0].plot()
    #         cv2.imshow('frame', frame)
    #         cv2.waitKey(10)
    #         num += 1
    #         out.write(frame)
    # except Exception as e:
    #     print(e)
    #     exit(1)
    # finally:
    #     cap.release()
    #     cv2.destroyAllWindows()
