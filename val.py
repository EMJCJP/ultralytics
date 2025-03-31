from torch.xpu import device

from ultralytics import YOLO
# 加载模型
model = YOLO("allsteel/bestn.pt")  # 加载自定义模型
# 验证模型
metrics = model.val(
    data="./data/all-steel.yaml",
    workers=0,
    batch=4,
    device=0,
    save_txt=True,
    conf=0.4,
    iou=0.5,
    # save_hybrid=True,
)  # 不需要参数,数据集和设置会被记住