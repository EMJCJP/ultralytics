
from ultralytics import YOLO

import os

if __name__ == '__main__':
    # model = YOLO('./seg/best.pt')

    model = YOLO('allsteel/best.pt')
    model.predict(source=r'D:\DataSets\XRay\full\zawu_bubble\test',
                imgsz=640,
                device='0',
                save=True,
                save_txt=True,
                )


    # model = YOLO('runs/segment/train18/weights/best.pt')
    # model.predict(source=r'D:\DataSets\Seg\test',
    #             imgsz=640,
    #             device='0',
    #             save=True,
    #             show_boxes=False,
    #             show_labels=False,
    #             save_txt=True,
    #             )


    # dir = r"C:\Users\Administrator\Desktop\test"
    # for file in os.listdir(dir):
    #     results = model(os.path.join(dir, file))
    #     # result_image = results[0].masks
    #     result_image = results[0].plot(boxes=False, show=True, save=True, filename=os.path.join(dir, file).split('.')[0] + "_seg.jpg")  # 设置boxes=False来隐藏边界框
    # # results = model(r"D:\Cursor Project\temp\3_2.jpg")
    # # # result_image = results[0].masks
    # # result_image = results[0].plot(boxes=False, show=True, save=True, filename=r"D:\Cursor Project\3_2_seg.jpg")  # 设置boxes=False来隐藏边界框
    #
    # # # 显示或保存结果
    # # results[0].show()  # 显示结果
    # # results[0].save(filename=r"D:\Cursor Project\1_1.jpg")  # 保存结果
    #
    # # D:\DataSets\XRay\single_curve\new_curve\flip\curve\images\val
    # # D:\DataSets\XRay\3in1\zawu_bubble\zawu\images\val
    # # D:\DataSets\XRay\y\half_data\images\val