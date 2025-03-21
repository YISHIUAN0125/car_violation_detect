from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    dev = torch.device('cuda:0' if torch.   cuda.is_available() else 'cpu')
    print("開始訓練 .........") 
    model = YOLO("yolo11n.pt")
    results = model.train(data=r"E:\NCU\yolov11\data.yaml", batch=25 , epochs=400, imgsz=640, device=dev, patience=30)
    # path=model.export()
    path = model.export(format='ncnn', imgsz = 640)
    print(f'模型匯出路徑 : {path}')