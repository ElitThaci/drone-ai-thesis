from ultralytics import YOLO
import time, shutil

model = YOLO("yolov8n.pt")

print("Exporting FP32...")
t0 = time.time()
model.export(format="engine", imgsz=640, half=False, device=0, workspace=4)
print(f"FP32 done in {time.time()-t0:.0f}s")
shutil.move("yolov8n.engine", "yolov8n_fp32.engine")

print("Exporting FP16...")
t0 = time.time()
model.export(format="engine", imgsz=640, half=True, device=0, workspace=4)
print(f"FP16 done in {time.time()-t0:.0f}s")
shutil.move("yolov8n.engine", "yolov8n_fp16.engine")

print("Exporting INT8...")
t0 = time.time()
model.export(format="engine", imgsz=640, int8=True, device=0, workspace=4, data="coco8.yaml")
print(f"INT8 done in {time.time()-t0:.0f}s")
shutil.move("yolov8n.engine", "yolov8n_int8.engine")

print("\nTë gjitha engines u eksportuan:")
print(" - yolov8n_fp32.engine")
print(" - yolov8n_fp16.engine")
print(" - yolov8n_int8.engine")
