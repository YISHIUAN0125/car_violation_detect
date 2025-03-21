from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob

if __name__=='__main__':
    classes = []
    sources = r'.\0816_record\redline'
    model_path=r'runs\detect\train\weights\best_ncnn_model'
    result_path = r'.\predicted'
    os.makedirs(result_path, exist_ok=True)
    model = YOLO(model_path)
    jpg_file = glob.glob(os.path.join(sources, '*.jpg'))
    for file in jpg_file:
        img = cv2.imread(file)
        results = model.predict(img, conf = 0.5)
        boxes = results[0].boxes #get boxes
        for box in boxes:
            id = int(box.cls)
            print(box.cls.item()) # class id
            print(box.xyxy)  # box coordinates (tensor)
            print(f'accuracy : {box.conf.item()}') # confidence 
            xy_arr = box.xyxy.cpu()
            coordi = np.array(xy_arr)
            x_mid = (coordi[:, 0] + coordi[:, 2]) / 2
            y_mid = (coordi[:, 1] + coordi[:, 3]) / 2
            midpoints = np.column_stack((x_mid, y_mid)) #middle point of the rectangular
            print(f'mid point(x,y) : { midpoints}')
            print(f'x_1,y_1 : { coordi[:, 0]},{coordi[:, 1]}')
            print(f'x_2,y_2 : { coordi[:, 2]},{coordi[:, 3]}')
            print(f'mid point(x,y) : { midpoints}')
        annotaionImg = results[0].plot()
        cv2.imshow('test', annotaionImg)
        cv2.waitKey(0)
