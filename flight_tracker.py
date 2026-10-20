import cv2, csv, time, os, struct, signal
from ultralytics import YOLO
from pid_controller import PIDController
from performance_logger import PerformanceLogger
from sahi_inference import SAHIDetector

# ─── SAHI toggle signal ───────────────────────────────────────
SAHI_MODE = False

def toggle_sahi(signum, frame_sig):
    global SAHI_MODE
    SAHI_MODE = not SAHI_MODE
    if SAHI_MODE:
        perf_logger.stop()
        print(f"\nSAHI mode: ON", flush=True)
    else:
        perf_logger.start()
        print(f"\nSAHI mode: OFF", flush=True)

signal.signal(signal.SIGUSR1, toggle_sahi)
print("Per SAHI toggle: kill -SIGUSR1 $(pgrep -f flight_tracker.py)")

# ─── FC serial ───────────────────────────────────────────────
try:
    import serial
    ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=0.1)
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

CLASS_PRIORITY = {0: 0, 2: 1, 5: 1, 7: 1}
CLASS_NAMES    = {0: "person", 2: "car", 5: "bus", 7: "truck"}
CLASSES        = list(CLASS_PRIORITY.keys())

RC_CENTER      = 1500
RC_MIN         = 1300
RC_MAX         = 1700
DEAD_ZONE_X    = 80
DEAD_ZONE_Y    = 100
AREA_DEAD_ZONE = 8000  # nuk reagon nese ndryshimi i areas eshte i vogel

CONF_NORMAL     = 0.35
CONF_AGGRESSIVE = 0.55
THERMAL_LIMIT   = 75.0

# ─── PID controllers ─────────────────────────────────────────
pid_yaw   = PIDController(kp=0.5,  ki=0.01, kd=0.05,
                           output_min=-150,  output_max=150)
pid_pitch = PIDController(kp=0.15, ki=0.003, kd=0.01,
                           output_min=-30,  output_max=30)

# ─── target selection ─────────────────────────────────────────
def select_target(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    best_idx      = 0
    best_priority = 999
    best_dist     = float('inf')
    for i, (box, cls) in enumerate(zip(boxes.xyxy, boxes.cls)):
        cls_int  = int(cls.item())
        priority = CLASS_PRIORITY.get(cls_int, 99)
        cx   = (box[0] + box[2]) / 2
        cy   = (box[1] + box[3]) / 2
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
    cv2.line(frame, (CENTER_X-30, CENTER_Y),
                    (CENTER_X+30, CENTER_Y), c, 2)
    cv2.line(frame, (CENTER_X, CENTER_Y-30),
                    (CENTER_X, CENTER_Y+30), c, 2)
    cv2.circle(frame, (CENTER_X, CENTER_Y), 8, c, 2)

def draw_target_info(frame, tx, ty, err_x, err_y,
                     rc_yaw, rc_pitch, tid,
                     cls_name, box_area, target_area):
    c = (0, 255, 255)
    cv2.line(frame, (CENTER_X, CENTER_Y), (tx, ty), c, 1)
    cv2.putText(frame,
        f"Err X:{err_x:+d} Area:{box_area} Target:{target_area}",
        (10, FRAME_H-80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, c, 2)
    cv2.putText(frame,
        f"Yaw:{rc_yaw} Pitch:{rc_pitch} Thr:MANUAL",
        (10, FRAME_H-50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    cv2.putText(frame,
        f"TARGET {cls_name} ID:{tid}",
        (tx+10, ty-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

def draw_status(frame, perf, fc, n_obj, fps,
                lost, conf, sahi_mode, area_set):
    ram = perf["ram_percent"]
    gpu = perf["gpu_temp_c"]
    pwr = perf["power_mw"]
    fc_str    = "FC:OK" if fc else "FC:SIM"
    lost_col  = (0,0,255) if lost else (255,255,255)
    sahi_str  = " [SAHI]" if sahi_mode else " [NORMAL]"
    area_str  = " DIST:SET" if area_set else " DIST:PENDING"
    area_col  = (0,255,0) if area_set else (0,165,255)
    cv2.putText(frame,
        f"{fc_str} Obj:{n_obj} FPS:{fps:.1f}{sahi_str}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lost_col, 2)
    cv2.putText(frame,
        f"GPU:{gpu:.1f}C RAM:{ram}% PWR:{pwr}mW",
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255,255,255), 2)
    cv2.putText(frame,
        f"YAW+PITCH:AI  THR:MANUAL{area_str}",
        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
        area_col, 2)

# ─── ngarko modelet ───────────────────────────────────────────
print("Duke ngarkuar modelet...")
model_fp16 = YOLO("yolov8n_fp16.engine", task="detect")
model_sahi = YOLO("yolov8n_fp16.engine", task="detect")
sahi_det   = SAHIDetector(model_sahi,
                           slice_w=640, slice_h=360,
                           overlap=0.2, conf=0.35)
print("Modelet u ngarkuan.")

# ─── performance logger + kamera ─────────────────────────────
perf_logger = PerformanceLogger()
perf_logger.start()

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
    "x1", "y1", "x2", "y2", "conf", "box_area",
    "target_cx", "target_cy", "err_x", "err_y",
    "rc_yaw", "rc_pitch",
    "conf_threshold", "sahi_mode",
    "ram_pct", "gpu_temp", "cpu_temp", "power_mw",
    "latency_ms", "fps", "target_lost", "target_area"
])

print(f"Saving to : {save_dir}")
print(f"FC        : {'CONNECTED' if FC_CONNECTED else 'SIMULATION'}")
print("Ctrl+C to stop\n")

frame_count     = 0
prev_yaw        = RC_CENTER
prev_pitch      = RC_CENTER
TARGET_AREA     = 0
target_area_set = False

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ─── performance ──────────────────────────────────────
        perf        = perf_logger.get()
        is_throttle = perf_logger.is_throttle_needed(THERMAL_LIMIT)
        gpu_freq    = perf.get("gpu_freq_pct", 0)
        conf_thr    = CONF_AGGRESSIVE if gpu_freq > 80 else CONF_NORMAL

        # ─── inference ────────────────────────────────────────
        t0          = time.perf_counter()
        n_objects   = 0
        rc_yaw      = RC_CENTER
        rc_pitch    = RC_CENTER
        target_lost = True
        lat         = 0.0
        fps         = 0.0

        annotated = frame.copy()
        draw_crosshair(annotated)

        if SAHI_MODE:
            dets      = sahi_det.detect(frame, classes=CLASSES)
            lat       = (time.perf_counter() - t0) * 1000
            fps       = 1000 / lat if lat > 0 else 0
            n_objects = len(dets)

            if dets:
                best_idx  = 0
                best_dist = float('inf')
                for i, det in enumerate(dets):
                    cx = (det['box'][0] + det['box'][2]) / 2
                    cy = (det['box'][1] + det['box'][3]) / 2
                    d  = ((cx-CENTER_X)**2+(cy-CENTER_Y)**2)**0.5
                    if d < best_dist:
                        best_dist = d
                        best_idx  = i

                for i, det in enumerate(dets):
                    x1,y1,x2,y2 = [int(v) for v in det['box']]
                    cls_name = CLASS_NAMES.get(
                        det['cls'], str(det['cls']))
                    cx       = (x1+x2)//2
                    cy       = (y1+y2)//2
                    box_area = (x2-x1) * (y2-y1)
                    col = (0,255,255) if i==best_idx else (0,255,0)
                    cv2.rectangle(annotated,
                        (x1,y1),(x2,y2), col, 2)
                    cv2.putText(annotated,
                        f"{cls_name} {det['conf']:.2f}",
                        (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                    if i == best_idx:
                        target_lost = False

                        # vendos target area automatikisht
                        if not target_area_set:
                            TARGET_AREA     = box_area
                            target_area_set = True
                            print(f"TARGET_AREA vendosur: {TARGET_AREA}")

                        err_x    = cx - CENTER_X
                        err_y    = cy - CENTER_Y
                        area_err = TARGET_AREA - box_area

                        eff_x    = err_x if abs(err_x)>DEAD_ZONE_X else 0
                        eff_area = area_err if abs(area_err)>AREA_DEAD_ZONE else 0

                        yaw_out   = pid_yaw.compute(eff_x)
                        pitch_out = pid_pitch.compute(eff_area)

                        rc_yaw   = int(max(RC_MIN,min(RC_MAX,
                                       RC_CENTER+yaw_out)))
                        rc_pitch = int(max(RC_MIN,min(RC_MAX,
                                       RC_CENTER+pitch_out)))

                        draw_target_info(annotated, cx, cy,
                            err_x, err_y, rc_yaw, rc_pitch,
                            i, cls_name, box_area, TARGET_AREA)

                        if FC_CONNECTED and ser:
                            rc_yaw = int(max(prev_yaw-50,
                                         min(prev_yaw+50, rc_yaw)))
                            rc_pitch = int(max(prev_pitch-10,
                                           min(prev_pitch+10, rc_pitch)))
                            prev_yaw   = rc_yaw
                            prev_pitch = rc_pitch
                            send_msp_rc(ser, [
                                RC_CENTER,  # roll — manual
                                rc_pitch,   # pitch — Jetson
                                RC_CENTER,  # throttle — MANUAL
                                rc_yaw,     # yaw — Jetson
                                2000,       # AUX1
                                RC_CENTER,
                                RC_CENTER,
                                RC_CENTER,
                            ])

                        writer.writerow([
                            time.time(), frame_count,
                            i, cls_name, 0,
                            x1,y1,x2,y2,
                            round(det['conf'],3), box_area,
                            cx, cy, err_x, err_y,
                            rc_yaw, rc_pitch,
                            conf_thr, 1,
                            perf["ram_percent"],
                            perf["gpu_temp_c"],
                            perf["cpu_temp_c"],
                            perf["power_mw"],
                            round(lat,2), round(fps,1),
                            0, TARGET_AREA
                        ])

        else:
            results = model_fp16.track(
                frame, persist=True,
                tracker="bytetrack.yaml",
                conf=conf_thr, iou=0.45,
                classes=CLASSES, verbose=False
            )
            lat      = (time.perf_counter() - t0) * 1000
            fps      = 1000 / lat if lat > 0 else 0
            annotated = results[0].plot()
            draw_crosshair(annotated)

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
                    cls_name = CLASS_NAMES.get(
                        cls_int, str(cls_int))
                    priority = CLASS_PRIORITY.get(cls_int, 99)
                    cx       = (x1+x2)//2
                    cy       = (y1+y2)//2
                    box_area = (x2-x1) * (y2-y1)
                    err_x    = cx - CENTER_X
                    err_y    = cy - CENTER_Y
                    area_err = TARGET_AREA - box_area

                    if i == primary:
                        target_lost = False

                        # vendos target area automatikisht
                        if not target_area_set:
                            TARGET_AREA     = box_area
                            target_area_set = True
                            print(f"TARGET_AREA vendosur: {TARGET_AREA}")

                        eff_x    = err_x if abs(err_x)>DEAD_ZONE_X else 0
                        eff_area = area_err if abs(area_err)>AREA_DEAD_ZONE else 0

                        yaw_out   = pid_yaw.compute(eff_x)
                        pitch_out = pid_pitch.compute(eff_area)

                        rc_yaw   = int(max(RC_MIN,min(RC_MAX,
                                       RC_CENTER+yaw_out)))
                        rc_pitch = int(max(RC_MIN,min(RC_MAX,
                                       RC_CENTER+pitch_out)))

                        cv2.rectangle(annotated,
                            (x1,y1),(x2,y2),(0,255,255),3)
                        draw_target_info(annotated, cx, cy,
                            err_x, err_y, rc_yaw, rc_pitch,
                            tid_int, cls_name,
                            box_area, TARGET_AREA)

                        if FC_CONNECTED and ser:
                            rc_yaw = int(max(prev_yaw-20,
                                         min(prev_yaw+20, rc_yaw)))
                            rc_pitch = int(max(prev_pitch-10,
                                           min(prev_pitch+10, rc_pitch)))
                            prev_yaw   = rc_yaw
                            prev_pitch = rc_pitch
                            send_msp_rc(ser, [
                                RC_CENTER,  # roll — manual
                                rc_pitch,   # pitch — Jetson
                                RC_CENTER,  # throttle — MANUAL
                                rc_yaw,     # yaw — Jetson
                                2000,       # AUX1
                                RC_CENTER,
                                RC_CENTER,
                                RC_CENTER,
                            ])

                        writer.writerow([
                            time.time(), frame_count,
                            tid_int, cls_name, priority,
                            x1,y1,x2,y2,
                            round(conf.item(),3), box_area,
                            cx, cy, err_x, err_y,
                            rc_yaw, rc_pitch,
                            conf_thr, 0,
                            perf["ram_percent"],
                            perf["gpu_temp_c"],
                            perf["cpu_temp_c"],
                            perf["power_mw"],
                            round(lat,2), round(fps,1),
                            0, TARGET_AREA
                        ])

        # ─── target lost ──────────────────────────────────────
        if target_lost:
            pid_yaw.reset()
            pid_pitch.reset()
            prev_yaw   = RC_CENTER
            prev_pitch = RC_CENTER
            if FC_CONNECTED and ser:
                send_msp_rc(ser, [
                    RC_CENTER,  # roll — manual
                    RC_CENTER,  # pitch — neutral
                    RC_CENTER,  # throttle — MANUAL
                    RC_CENTER,  # yaw — neutral
                    2000,       # AUX1
                    RC_CENTER,
                    RC_CENTER,
                    RC_CENTER,
                ])
            writer.writerow([
                time.time(), frame_count,
                None,None,None,
                None,None,None,None,None,None,
                None,None,None,None,
                RC_CENTER, RC_CENTER,
                conf_thr, int(SAHI_MODE),
                perf["ram_percent"],
                perf["gpu_temp_c"],
                perf["cpu_temp_c"],
                perf["power_mw"],
                round(lat,2), round(fps,1),
                1, TARGET_AREA
            ])

        # ─── status + ruajtja ─────────────────────────────────
        draw_status(annotated, perf, FC_CONNECTED,
                    n_objects, fps, target_lost,
                    conf_thr, SAHI_MODE, target_area_set)
        out.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            mode_str = "SAHI  " if SAHI_MODE else "NORMAL"
            print(
                f"Frame:{frame_count:5d} | "
                f"Mode:{mode_str} | "
                f"FPS:{fps:5.1f} | "
                f"Obj:{n_objects} | "
                f"GPU:{perf['gpu_temp_c']:.1f}C | "
                f"Area:{TARGET_AREA} | "
                f"Lost:{target_lost}"
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
