import cv2, csv, time, os, struct
from ultralytics import YOLO
from pid_controller import PIDController
from performance_logger import PerformanceLogger

# ─── FC serial ───────────────────────────────────────────────
try:
    import serial
    ser = serial.Serial('/dev/ttyTHS0', 115200, timeout=0.1)
    FC_CONNECTED = True
    print("FC connected via MSP")
except Exception:
    ser = None
    FC_CONNECTED = False
    print("FC not connected — simulation mode")

# ─── MSP SET_RAW_RC ───────────────────────────────────────────
def send_msp_rc(ser, channels):
    payload = b''
    for ch in channels:
        payload += struct.pack('<H', ch)
    size = len(payload)
    cmd  = 200
    crc  = size ^ cmd
    for b in payload:
        crc ^= b
    ser.write(b'$M<' + bytes([size, cmd]) + payload + bytes([crc]))

# ─── GStreamer ────────────────────────────────────────────────
def gst_pipeline(w=1280, h=720, fps=30):
    return (
        f"nvarguscamerasrc sensor-id=0 ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},"
        f"framerate={fps}/1,format=NV12 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw,width={w},height={h},format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    )

# ─── config ──────────────────────────────────────────────────
FRAME_W  = 1280
FRAME_H  = 720
CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

# classes: person ka prioritet 0 (me te larte), vehicle 1
CLASS_PRIORITY = {
    0: 0,  # person   — prioritet 1
    2: 1,  # car      — prioritet 2
    5: 1,  # bus      — prioritet 2
    7: 1,  # truck    — prioritet 2
}
CLASS_NAMES = {0: "person", 2: "car", 5: "bus", 7: "truck"}
CLASSES = list(CLASS_PRIORITY.keys())

RC_CENTER    = 1500
RC_MIN       = 1300
RC_MAX       = 1700
DEAD_ZONE_X  = 40
DEAD_ZONE_Y  = 30

# confidence thresholds
CONF_NORMAL     = 0.35
CONF_AGGRESSIVE = 0.55
THERMAL_LIMIT   = 75.0   # celsius — above this throttle to normal inference

# ─── PID ─────────────────────────────────────────────────────
pid_yaw      = PIDController(kp=0.6, ki=0.03, kd=0.08,
                              output_min=-200, output_max=200)
pid_throttle = PIDController(kp=0.4, ki=0.02, kd=0.05,
                              output_min=-150, output_max=150)

# ─── target selection ─────────────────────────────────────────
def select_target(boxes):
    """
    Zgjedh targetin me prioritet me te larte (person > vehicle)
    nese ka te njejtin prioritet, zgjedh me afert crosshair-it
    """
    if boxes is None or len(boxes) == 0:
        return None

    best_idx      = 0
    best_priority = 999
    best_dist     = float('inf')

    for i, (box, cls) in enumerate(zip(boxes.xyxy, boxes.cls)):
        cls_int  = int(cls.item())
        priority = CLASS_PRIORITY.get(cls_int, 99)
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        dist = ((cx - CENTER_X)**2 + (cy - CENTER_Y)**2) ** 0.5

        if priority < best_priority:
            best_priority = priority
            best_dist     = dist
            best_idx      = i
        elif priority == best_priority and dist < best_dist:
            best_dist = dist
            best_idx  = i

    return best_idx

# ─── drawing ──────────────────────────────────────────────────
def draw_crosshair(frame):
    c = (0, 255, 255)
    cv2.line(frame, (CENTER_X-30, CENTER_Y), (CENTER_X+30, CENTER_Y), c, 2)
    cv2.line(frame, (CENTER_X, CENTER_Y-30), (CENTER_X, CENTER_Y+30), c, 2)
    cv2.circle(frame, (CENTER_X, CENTER_Y), 8, c, 2)

def draw_target_info(frame, tx, ty, err_x, err_y,
                     rc_yaw, rc_thr, tid, cls_name):
    c = (0, 255, 255)
    cv2.line(frame, (CENTER_X, CENTER_Y), (tx, ty), c, 1)
    cv2.putText(frame, f"Err X:{err_x:+d} Y:{err_y:+d}",
                (10, FRAME_H-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    cv2.putText(frame, f"Yaw:{rc_yaw} Thr:{rc_thr}",
                (10, FRAME_H-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"TARGET {cls_name} ID:{tid}",
                (tx+10, ty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

def draw_status(frame, perf, fc, n_obj, fps,
                lost, conf, throttling):
    ram  = perf["ram_percent"]
    gpu  = perf["gpu_temp_c"]
    cpu  = perf["cpu_temp_c"]
    pwr  = perf["power_mw"]

    fc_str   = "FC:OK" if fc else "FC:SIM"
    thr_str  = " THERMAL-THROTTLE" if throttling else ""
    lost_str = " TARGET-LOST" if lost else ""
    col_lost = (0,0,255) if lost else (255,255,255)
    col_thr  = (0,165,255) if throttling else (255,255,255)

    cv2.putText(frame,
        f"{fc_str} Obj:{n_obj} FPS:{fps:.1f} CONF:{conf:.2f}{lost_str}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_lost, 2)
    cv2.putText(frame,
        f"GPU:{gpu:.1f}C CPU:{cpu:.1f}C RAM:{ram}% PWR:{pwr}mW{thr_str}",
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_thr, 2)

# ─── main ────────────────────────────────────────────────────
perf_logger = PerformanceLogger()
perf_logger.start()

model_fp16 = YOLO("yolov8n_fp16.engine", task="detect")
model_fp32 = YOLO("yolov8n_fp32.engine", task="detect")

cap = cv2.VideoCapture(gst_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("ERROR: camera failed to open")
    exit()

session  = time.strftime("%Y%m%d_%H%M%S")
save_dir = os.path.expanduser(f"~/flight_data/{session}")
os.makedirs(save_dir, exist_ok=True)

out = cv2.VideoWriter(
    f"{save_dir}/video.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    30, (FRAME_W, FRAME_H)
)

log    = open(f"{save_dir}/tracking.csv", "w", newline="")
writer = csv.writer(log)
writer.writerow([
    "timestamp", "frame", "track_id", "class", "priority",
    "x1", "y1", "x2", "y2", "conf",
    "target_cx", "target_cy", "err_x", "err_y",
    "rc_yaw", "rc_throttle",
    "conf_threshold", "thermal_throttle",
    "ram_pct", "gpu_temp", "cpu_temp", "power_mw",
    "latency_ms", "fps", "target_lost"
])

print(f"Saving to : {save_dir}")
print(f"FC        : {'CONNECTED' if FC_CONNECTED else 'SIMULATION'}")
print("Ctrl+C to stop\n")

frame_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ─── thermal throttling ───────────────────────────────
        perf         = perf_logger.get()
        is_throttle  = perf_logger.is_throttle_needed(THERMAL_LIMIT)
        active_model = model_fp32 if is_throttle else model_fp16

        # ─── adaptive confidence ──────────────────────────────
        # simulojme lëvizje agresive me GPU load si proxy
        # (do zevendesohet me roll/pitch nga FC kur droni te jete gati)
        gpu_freq = perf.get("gpu_freq_pct", 0)
        conf_thr = CONF_AGGRESSIVE if gpu_freq > 80 else CONF_NORMAL

        # ─── inference ────────────────────────────────────────
        t0 = time.perf_counter()
        results = active_model.track(
            frame, persist=True,
            tracker="bytetrack.yaml",
            conf=conf_thr,
            iou=0.45,
            classes=CLASSES,
            verbose=False
        )
        lat = (time.perf_counter() - t0) * 1000
        fps = 1000 / lat

        annotated   = results[0].plot()
        draw_crosshair(annotated)

        n_objects   = 0
        rc_yaw      = RC_CENTER
        rc_throttle = RC_CENTER
        target_lost = True

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes
            n_objects = len(boxes)
            primary   = select_target(boxes)

            for i, (box, tid, conf, cls) in enumerate(zip(
                boxes.xyxy, boxes.id,
                boxes.conf, boxes.cls
            )):
                x1,y1,x2,y2 = [int(v) for v in box.tolist()]
                tid_int  = int(tid.item())
                cls_int  = int(cls.item())
                cls_name = CLASS_NAMES.get(cls_int, str(cls_int))
                priority = CLASS_PRIORITY.get(cls_int, 99)
                cx       = (x1+x2)//2
                cy       = (y1+y2)//2
                err_x    = cx - CENTER_X
                err_y    = cy - CENTER_Y

                if i == primary:
                    target_lost = False

                    eff_x = err_x if abs(err_x) > DEAD_ZONE_X else 0
                    eff_y = err_y if abs(err_y) > DEAD_ZONE_Y else 0

                    yaw_out = pid_yaw.compute(eff_x)
                    thr_out = pid_throttle.compute(-eff_y)

                    rc_yaw      = int(max(RC_MIN, min(RC_MAX,
                                      RC_CENTER + yaw_out)))
                    rc_throttle = int(max(RC_MIN, min(RC_MAX,
                                      RC_CENTER + thr_out)))

                    cv2.rectangle(annotated,
                        (x1,y1),(x2,y2),(0,255,255),3)
                    draw_target_info(annotated, cx, cy,
                        err_x, err_y, rc_yaw, rc_throttle,
                        tid_int, cls_name)

                    if FC_CONNECTED and ser:
                        send_msp_rc(ser, [
                            RC_CENTER, RC_CENTER,
                            rc_throttle, rc_yaw,
                            RC_CENTER, RC_CENTER,
                            RC_CENTER, RC_CENTER
                        ])

                    writer.writerow([
                        time.time(), frame_count,
                        tid_int, cls_name, priority,
                        x1,y1,x2,y2,
                        round(conf.item(),3),
                        cx, cy, err_x, err_y,
                        rc_yaw, rc_throttle,
                        conf_thr, int(is_throttle),
                        perf["ram_percent"],
                        perf["gpu_temp_c"],
                        perf["cpu_temp_c"],
                        perf["power_mw"],
                        round(lat,2), round(fps,1), 0
                    ])

        if target_lost:
            pid_yaw.reset()
            pid_throttle.reset()
            if FC_CONNECTED and ser:
                send_msp_rc(ser, [RC_CENTER]*8)
            writer.writerow([
                time.time(), frame_count,
                None, None, None,
                None,None,None,None,None,
                None,None,None,None,
                RC_CENTER, RC_CENTER,
                conf_thr, int(is_throttle),
                perf["ram_percent"],
                perf["gpu_temp_c"],
                perf["cpu_temp_c"],
                perf["power_mw"],
                round(lat,2), round(fps,1), 1
            ])

        draw_status(annotated, perf, FC_CONNECTED,
                    n_objects, fps, target_lost,
                    conf_thr, is_throttle)
        out.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            print(
                f"Frame:{frame_count:5d} | "
                f"FPS:{fps:5.1f} | "
                f"Obj:{n_objects} | "
                f"GPU:{perf['gpu_temp_c']:.1f}C | "
                f"RAM:{perf['ram_percent']}% | "
                f"Conf:{conf_thr:.2f} | "
                f"Throttle:{is_throttle}"
            )

except KeyboardInterrupt:
    print("\nStopped")

finally:
    cap.release()
    out.release()
    log.close()
    perf_logger.stop()
    if ser:
        ser.close()
    print(f"Saved to {save_dir}")
