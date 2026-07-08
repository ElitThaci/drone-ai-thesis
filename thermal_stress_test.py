import time, csv, subprocess, re, threading
from ultralytics import YOLO
import numpy as np

# ─── lexues i temperaturës në thread të veçantë ───────────────
hw_data = {"gpu_temp": 0.0, "cpu_temp": 0.0, "ram_pct": 0.0, "power_mw": 0}
def read_hw():
    proc = subprocess.Popen(
        ["tegrastats", "--interval", "1000"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    for line in proc.stdout:
        try:
            gpu_t = re.search(r'gpu@([\d.]+)C', line)
            cpu_t = re.search(r'cpu@([\d.]+)C', line)
            ram_m = re.search(r'RAM (\d+)/(\d+)MB', line)
            pwr_m = re.search(r'VDD_IN (\d+)mW', line)
            if gpu_t: hw_data["gpu_temp"]  = float(gpu_t.group(1))
            if cpu_t: hw_data["cpu_temp"]  = float(cpu_t.group(1))
            if ram_m: hw_data["ram_pct"]   = round(
                int(ram_m.group(1))/int(ram_m.group(2))*100, 1)
            if pwr_m: hw_data["power_mw"]  = int(pwr_m.group(1))
        except Exception:
            pass

hw_thread = threading.Thread(target=read_hw, daemon=True)
hw_thread.start()
time.sleep(2)  # lejo thread-in të lexojë të dhënat e para

# ─── model + log ──────────────────────────────────────────────
model = YOLO("yolov8n_fp16.engine", task="detect")
dummy = (255 * np.random.rand(720, 1280, 3)).astype("uint8")

log    = open("/home/dnja/thermal_stress_log.csv", "w", newline="")
writer = csv.writer(log)
writer.writerow(["time_s","fps","latency_ms",
                 "gpu_temp","cpu_temp","ram_pct","power_mw"])

print("Thermal stress test — 30 minuta. Ctrl+C per nderprerje.")
start   = time.perf_counter()
last_print = 0
while True:
    t0  = time.perf_counter()
    model(dummy, device=0, verbose=False)
    lat = (time.perf_counter() - t0) * 1000
    fps = 1000 / lat
    elapsed = time.perf_counter() - start

    writer.writerow([
        round(elapsed, 1), round(fps, 1), round(lat, 2),
        hw_data["gpu_temp"], hw_data["cpu_temp"],
        hw_data["ram_pct"],  hw_data["power_mw"]
    ])


    if elapsed - last_print >= 60:
        last_print = elapsed
        print(f"  t={elapsed/60:.1f}min | FPS:{fps:.1f} | "
              f"GPU:{hw_data['gpu_temp']}C | "
              f"CPU:{hw_data['cpu_temp']}C | "
              f"RAM:{hw_data['ram_pct']}% | "
              f"PWR:{hw_data['power_mw']}mW")

    if elapsed >= 1800:
        print("Test i perfunduar!")
        break

log.close()
print("Rezultatet: ~/thermal_stress_log.csv")
