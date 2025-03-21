from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"runs\detect\train29\weights\best.pt")
    path = model.export(format='ncnn', int8 = True, half = True, nms = True, imgsz = 640)