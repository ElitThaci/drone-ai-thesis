import numpy as np
import time
import csv
from ultralytics import YOLO

ENGINES = {
    "FP32": "/home/dnja/yolov8n_fp32.engine",
    "FP16": "/home/dnja/yolov8n_fp16.engine",
    "INT8": "/home/dnja/yolov8n_int8.engine",
}

WARMUP_RUNS = 5
TEST_RUNS   = 100
IMG_W, IMG_H = 1280, 720

results = []

for name, path in ENGINES.items():
    print(f"\nTestoj {name} engine...")
    model = YOLO(path, task="detect")

    dummy = (255 * np.random.rand(IMG_H, IMG_W, 3)).astype("uint8")

    # warmup
    for _ in range(WARMUP_RUNS):
        model(dummy, device=0, verbose=False)

    # matje
    latencies = []
    detections = []

    for _ in range(TEST_RUNS):
        t0 = time.perf_counter()
        r = model(dummy, device=0, verbose=False)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        detections.append(len(r[0].boxes))

    avg_lat = np.mean(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    std_lat = np.std(latencies)
    avg_fps = 1000 / avg_lat

    print(f"  Avg latency : {avg_lat:.1f} ms")
    print(f"  Min latency : {min_lat:.1f} ms")
    print(f"  Max latency : {max_lat:.1f} ms")
    print(f"  Std dev     : {std_lat:.1f} ms")
    print(f"  Avg FPS     : {avg_fps:.1f}")

    results.append({
        "engine": name,
        "avg_latency_ms": round(avg_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "std_ms":         round(std_lat, 2),
        "avg_fps":        round(avg_fps, 1),
    })

    del model

# ruaj rezultatet
with open("/home/dnja/benchmark_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n=== REZULTATET FINALE ===")
print(f"{'Engine':<8} {'Avg ms':<10} {'Min ms':<10} {'Max ms':<10} {'Std':<8} {'FPS'}")
print("-" * 55)
for r in results:
    print(f"{r['engine']:<8} {r['avg_latency_ms']:<10} {r['min_latency_ms']:<10} {r['max_latency_ms']:<10} {r['std_ms']:<8} {r['avg_fps']}")

print("\nRezultatet u ruajten ne: ~/benchmark_results.csv")
