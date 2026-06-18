import numpy as np
import time
import csv
from ultralytics import YOLO
from sahi_inference import SAHIDetector

MODEL_PATH = "/home/dnja/yolov8n_fp16.engine"
CLASSES    = [0, 2, 5, 7]
RUNS       = 30
WARMUP     = 3

model = YOLO(MODEL_PATH, task="detect")
sahi  = SAHIDetector(model, slice_w=640, slice_h=360, overlap=0.2)
dummy = (255 * np.random.rand(720, 1280, 3)).astype("uint8")

results = []

# ─── Normal inference ─────────────────────────────────────────
print("Testoj Normal Inference...")
for _ in range(WARMUP):
    model(dummy, device=0, verbose=False)

times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    r  = model(dummy, device=0, classes=CLASSES, verbose=False)
    times.append((time.perf_counter()-t0)*1000)

avg = np.mean(times)
results.append({
    "method":        "Normal",
    "patches":       1,
    "avg_ms":        round(avg, 2),
    "min_ms":        round(np.min(times), 2),
    "max_ms":        round(np.max(times), 2),
    "std_ms":        round(np.std(times), 2),
    "avg_fps":       round(1000/avg, 1),
})
print(f"  Normal: {avg:.1f}ms / {1000/avg:.1f}FPS")

# ─── SAHI konfigurime te ndryshme ─────────────────────────────
configs = [
    {"slice_w": 640, "slice_h": 360, "overlap": 0.1, "label": "SAHI_6patch"},
    {"slice_w": 640, "slice_h": 360, "overlap": 0.2, "label": "SAHI_9patch"},
    {"slice_w": 480, "slice_h": 270, "overlap": 0.2, "label": "SAHI_12patch"},
]

for cfg in configs:
    print(f"Testoj {cfg['label']}...")
    s = SAHIDetector(model,
                     slice_w=cfg["slice_w"],
                     slice_h=cfg["slice_h"],
                     overlap=cfg["overlap"])
    n_patches = s.get_slice_count(1280, 720)

    # warmup
    for _ in range(WARMUP):
        s.detect(dummy, classes=CLASSES)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        s.detect(dummy, classes=CLASSES)
        times.append((time.perf_counter()-t0)*1000)

    avg = np.mean(times)
    results.append({
        "method":  cfg["label"],
        "patches": n_patches,
        "avg_ms":  round(avg, 2),
        "min_ms":  round(np.min(times), 2),
        "max_ms":  round(np.max(times), 2),
        "std_ms":  round(np.std(times), 2),
        "avg_fps": round(1000/avg, 1),
    })
    print(f"  {cfg['label']}: {avg:.1f}ms / {1000/avg:.1f}FPS ({n_patches} patches)")

# ─── ruaj dhe printo ──────────────────────────────────────────
with open("/home/dnja/benchmark_sahi.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n=== REZULTATET SAHI vs NORMAL ===")
print(f"{'Method':<14} {'Patches':<10} {'Avg ms':<10} {'FPS':<8} {'Std ms'}")
print("-" * 52)
for r in results:
    print(f"{r['method']:<14} {r['patches']:<10} {r['avg_ms']:<10} {r['avg_fps']:<8} {r['std_ms']}")

print("\nRezultatet u ruajtën në: ~/benchmark_sahi.csv")
