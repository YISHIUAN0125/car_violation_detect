import cv2
from ultralytics import YOLO
import numpy as np
import os
import glob

class Result:
    def __init__(self):
        pass
    def redline(self, img:cv2.typing.MatLike, bboxs:list=None) -> cv2.typing.MatLike:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks to detect red color in two ranges
        mask1 = cv2.inRange(img, lower_red1, upper_red1)
        mask2 = cv2.inRange(img, lower_red2, upper_red2)
        mask = mask1 | mask2
        if bboxs is not None:
            for bbox in bboxs:
                x0, y0, x1, y1 = bbox
                mask[y0:y1, x0:x1] = 0
        mask = cv2.erode(mask, (3,3), iterations=5)
        mask = cv2.dilate(mask, (5,5), iterations=10)
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = line[0, :]
                cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return img, mask
        
    def check_parking_violation(self,  red_mask, bboxs:list=None) -> bool:
        x1, y1, x2, y2 = bboxs
        # Define areas around the car to check for red lines
        above_area = red_mask[max(0, y1 - 10):y1, x1:x2]
        below_area = red_mask[y2:y2 + 10, x1:x2]

        # Check if red lines exist in these regions
        above_red = np.any(above_area > 0)
        below_red = np.any(below_area > 0)

        return (above_red or below_red)

    def yellow_net(self, img:cv2.typing.MatLike, bboxs:list=None) -> cv2.typing.MatLike:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 40, 40])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(img, lower_yellow, upper_yellow)
        if bboxs is not None:
            for bbox in bboxs:
                x0, y0, x1, y1 = bbox
                mask[y0:y1, x0:x1] = 0
        mask = cv2.erode(mask, (3,3), iterations=2)
        mask = cv2.dilate(mask, (5,5), iterations=10)
        return img, mask
    
    def check_yellow_net_violation(self,  yellow_mask, bboxs:list=None) -> bool:
        x1, y1, x2, y2 = bboxs
        above_area = yellow_mask[max(0, y1-10):y1, x1:x2]
        below_area = yellow_mask[y2:y2+10, x1:x2]
        left_area = yellow_mask[y1:y2, max(0, x1-10):x1]
        right_area = yellow_mask[y1:y2, x2:x2+10]

        above_yellow = np.any(above_area > 0)
        below_yellow = np.any(below_area > 0)
        left_yellow = np.any(left_area > 0)
        right_yellow = np.any(right_area > 0)

        #check if yellow lines exist at least three regions
        yellow_count = sum([above_yellow, below_yellow, left_yellow, right_yellow])

        return yellow_count >= 3

if __name__ == "__main__":
    model = YOLO(r'runs\detect\train\weights\best_ncnn_model')
    generator = Result()
    source = "img_source"
    if not os.path.exists(source):
        os.makedirs(source)

    output = "output"
    if not os.path.exists(source):
        os.makedirs(source)

    image_files = []
    image_files.extend(glob.glob(os.path.join(source, '*jpg')))

    try:
        for i, f in enumerate(image_files):
            car_bbox = []
            Flag = False
            img = cv2.imread(f)
            result = model.predict(img)
            boxes = result[0].boxes
            for box in boxes:
                xy_arr = box.xyxy.cpu().numpy()
                x1, y1, x2, y2 = map(int, xy_arr[0])
                car_bbox.append((x1, y1, x2, y2))
            
            yellow_mask = generator.yellow_net(img, car_bbox)
            # red_mask = generator.redline(img, car_bbox) #紅線就用這個
            for _, box in enumerate(car_bbox):
                yellow_net_violation = generator.check_yellow_net_violation(yellow_mask, box)
                if yellow_net_violation:
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                    cv2.putText(img, f'yellow grid violation', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    Flag = True
            if Flag:
                cv2.imwrite(os.path.joint(output, (f"{i}.jpg")), img)
    except Exception as e:
        print(e)
        exit(1)

