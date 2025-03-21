import os
import cv2

def h264ToMp4(sourcePath,tagPath):
    cap = cv2.VideoCapture(sourcePath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 取得影像寬度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')          # 設定影片的格式為 MP4V
    out = cv2.VideoWriter(tagPath, fourcc, 30.0, (width,  height))  # 產生空的影片
    if not cap:
        print('failed')
        exit()
    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

if __name__=='__main__':
    path = ['video0228_1114']
    for i in path:
        source = "E:/NCU//yolov11/"+ i +'.h264'
        target = "E:/NCU//yolov11/"+ i + '.mp4'
        h264ToMp4(source,target)
    print('complete')