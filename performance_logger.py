import subprocess
import re
import threading
import time

class PerformanceLogger:
    def __init__(self):
        self.data = {
            "ram_used_mb":   0,
            "ram_total_mb":  0,
            "ram_percent":   0,
            "gpu_temp_c":    0,
            "cpu_temp_c":    0,
            "gpu_freq_pct":  0,
            "power_mw":      0,
        }
        self._running = False
        self._thread  = None

    def _parse(self, line):
        try:
            # RAM
            ram = re.search(r'RAM (\d+)/(\d+)MB', line)
            if ram:
                used  = int(ram.group(1))
                total = int(ram.group(2))
                self.data["ram_used_mb"]  = used
                self.data["ram_total_mb"] = total
                self.data["ram_percent"]  = round(used / total * 100, 1)

            # GPU frequency
            gpu_freq = re.search(r'GR3D_FREQ (\d+)%', line)
            if gpu_freq:
                self.data["gpu_freq_pct"] = int(gpu_freq.group(1))

            # temperatures
            gpu_temp = re.search(r'gpu@([\d.]+)C', line)
            if gpu_temp:
                self.data["gpu_temp_c"] = float(gpu_temp.group(1))

            cpu_temp = re.search(r'cpu@([\d.]+)C', line)
            if cpu_temp:
                self.data["cpu_temp_c"] = float(cpu_temp.group(1))

            # total power
            power = re.search(r'VDD_IN (\d+)mW', line)
            if power:
                self.data["power_mw"] = int(power.group(1))

        except Exception:
            pass

    def _run(self):
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "500"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        while self._running:
            line = proc.stdout.readline()
            if line:
                self._parse(line)
        proc.terminate()

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        time.sleep(1)  # lejo kohe te lexoje 
        print("Performance Logger started")

    def stop(self):
        self._running = False

    def get(self):
        return self.data.copy()

    def is_throttle_needed(self, threshold_c=75.0):
        return self.data["gpu_temp_c"] >= threshold_c
