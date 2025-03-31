from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./model_yaml/yolo11n-seg.yaml').load("yolo11n.pt")
    model.train(data='./data/seg.yaml',
                imgsz=640,
                epochs=2,
                batch=4,
                workers=2,
                device='0',
                )