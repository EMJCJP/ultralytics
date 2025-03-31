from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./model_yaml/yolo11n.yaml')
    model.train(data='./data/curve.yaml',
                imgsz=640,
                epochs=2,
                batch=4,
                workers=2,
                device='0',
                pretrained='./weights/yolo11n.pt',
                mosaic=1,
                scale=0.1,
                # neg_dir="neg_img",  # 负样本文件夹
                # neg_num=2,  # 负样加入数,继续训练30轮—正数2,重新训练300轮—负数负1（负样本很多时推荐为-1）,快速压误检（100epoch内）填2，平常用-1
                )