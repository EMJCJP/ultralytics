from ultralytics import YOLO

# model = YOLO('curve/best.pt')
model = YOLO('../allsteel/bestl.pt')

model.export(format="onnx", dynamic=True)


# trtexec.exe   --onnx=curve/best.onnx  --saveEngine=curve/burr.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:16x3x640x640 --maxShapes=images:32x3x640x640
# trtexec.exe   --onnx=zawu_bubble/best.onnx  --saveEngine=zawu_bubble/zawu.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:16x3x640x640 --maxShapes=images:32x3x640x640
# trtexec.exe   --onnx=bubble/best.onnx  --saveEngine=bubble/bubble.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:16x3x640x640 --maxShapes=images:32x3x640x640
# trtexec.exe   --onnx=zawu/best.onnx  --saveEngine=zawu/zawu.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:16x3x640x640 --maxShapes=images:32x3x640x640


# python -m onnxsim seg/end2end.onnx seg/end2end2.onnx
# trtexec.exe   --onnx=seg/best.onnx --explicitBatch --saveEngine=seg/best.trt --workspace=1024 --best
# trtexec.exe   --onnx=seg/end2end_dyn.onnx  --saveEngine=seg/end2end_dyn.trt --fp16 --buildOnly --minShapes=input:1x3x512x512 --optShapes=input:4x3x512x512 --maxShapes=input:4x3x512x512

# trtexec.exe   --onnx=seg/end2end_segformer.onnx  --saveEngine=seg/end2end_segformer.trt --buildOnly --minShapes=input:1x3x512x512 --optShapes=input:4x3x512x512 --maxShapes=input:4x3x512x512
#