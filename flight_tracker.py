import cv2, csv, time, os, struct
from ultralytics import YOLO
from pid_controller import PIDController

# ─── FC serial connection ─────────────────────────────────────
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

# ─── GStreamer pipeline ───────────────────────────────────────
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
FRAME_W    = 1280
FRAME_H    = 720
CENTER_X   = FRAME_W // 2
CENTER_Y   = FRAME_H // 2
CLASSES    = [0, 2, 5, 7]  # person, car, bus, truck
RC_CENTER  = 1500
RC_MIN     = 1300
RC_MAX     = 1700

# dead zone — pixels from center where no correction applied
# avoids micro-corrections when target is already nearly centered
DEAD_ZONE_X = 40
DEAD_ZONE_Y = 30

# ─── PID controllers ─────────────────────────────────────────
# yaw — horizontal centering
pid_yaw      = PIDController(
    kp=0.6, ki=0.03, kd=0.08,
    output_min=-200, output_max=200)

# throttle — vertical centering
pid_throttle = PIDController(
    kp=0.4, ki=0.02, kd=0.05,
    output_min=-150, output_max=150)

# ─── target selection ─────────────────────────────────────────
def select_target(boxes):
    """
    Select the object closest to frame center (crosshair).
    Returns index of selected target.
    """
    if boxes is None or len(boxes) == 0:
        return None
    min_dist = float('inf')
    best_idx = 0
    for i, box in enumerate(boxes.xyxy):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        dist = ((cx - CENTER_X)**2 + (cy - CENTER_Y)**2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx

# ─── drawing helpers ──────────────────────────────────────────
def draw_crosshair(frame):
    size = 30
    c    = (0, 255, 255)
    cv2.line(frame,
             (CENTER_X - size, CENTER_Y),
             (CENTER_X + size, CENTER_Y), c, 2)
    cv2.line(frame,
             (CENTER_X, CENTER_Y - size),
             (CENTER_X, CENTER_Y + size), c, 2)
    cv2.circle(frame, (CENTER_X, CENTER_Y), 8, c, 2)

def draw_target_info(frame, tx, ty, err_x, err_y,
                     rc_yaw, rc_throttle, tid, cls_name):
    c = (0, 255, 255)
    # line from center to target
    cv2.line(frame, (CENTER_X, CENTER_Y), (tx, ty), c, 1)
    # error values
    cv2.putText(frame,
        f"Err  X:{err_x:+d}  Y:{err_y:+d}",
        (10, FRAME_H - 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    # RC output values
    cv2.putText(frame,
        f"Yaw:{rc_yaw}  Thr:{rc_throttle}",
        (10, FRAME_H - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # target label
    cv2.putText(frame,
        f"TARGET  {cls_name}  ID:{tid}",
        (tx + 10, ty - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

def draw_status(frame, fc, n_obj, fps, lost):
    fc_str   = "FC:CONNECTED" if fc else "FC:SIMULATION"
    lost_str = "  TARGET LOST" if lost else ""
    color    = (0, 0, 255) if lost else (255, 255, 255)
    cv2.putText(frame,
        f"{fc_str}  Obj:{n_obj}  FPS:{fps:.1f}{lost_str}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ─── class id to name ─────────────────────────────────────────
CLASS_NAMES = {0: "person", 2: "car", 5: "bus", 7: "truck"}

# ─── main ────────────────────────────────────────────────────
model = YOLO("yolov8n.engine", task="detect")
cap   = cv2.VideoCapture(gst_pipeline(), cv2.CAP_GSTREAMER)

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
    "timestamp", "frame",
    "track_id", "class",
    "x1", "y1", "x2", "y2", "conf",
    "target_cx", "target_cy",
    "err_x", "err_y",
    "rc_yaw", "rc_throttle",
    "target_lost", "latency_ms"
])

print(f"Saving to : {save_dir}")
print(f"FC status : {'CONNECTED' if FC_CONNECTED else 'SIMULATION'}")
print("Ctrl+C to stop\n")

frame_count  = 0
target_lost  = False

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.35,
            iou=0.45,
            classes=CLASSES,
            verbose=False
        )
        lat = (time.perf_counter() - t0) * 1000
        fps = 1000 / lat

        annotated    = results[0].plot()
        draw_crosshair(annotated)

        n_objects   = 0
        rc_yaw      = RC_CENTER
        rc_throttle = RC_CENTER
        target_lost = True

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes
            n_objects = len(boxes)

            # select target closest to crosshair
            primary_idx = select_target(boxes)

            for i, (box, tid, conf, cls) in enumerate(zip(
                boxes.xyxy,
                boxes.id,
                boxes.conf,
                boxes.cls
            )):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                tid_int  = int(tid.item())
                cls_int  = int(cls.item())
                cls_name = CLASS_NAMES.get(cls_int, str(cls_int))
                cx       = (x1 + x2) // 2
                cy       = (y1 + y2) // 2
                err_x    = cx - CENTER_X
                err_y    = cy - CENTER_Y

                if i == primary_idx:
                    target_lost = False

                    # apply dead zone
                    eff_err_x = err_x if abs(err_x) > DEAD_ZONE_X else 0
                    eff_err_y = err_y if abs(err_y) > DEAD_ZONE_Y else 0

                    # PID compute
                    yaw_out      = pid_yaw.compute(eff_err_x)
                    throttle_out = pid_throttle.compute(-eff_err_y)

                    rc_yaw = int(max(RC_MIN,
                                 min(RC_MAX,
                                 RC_CENTER + yaw_out)))
                    rc_throttle = int(max(RC_MIN,
                                      min(RC_MAX,
                                      RC_CENTER + throttle_out)))

                    # highlight primary target
                    cv2.rectangle(annotated,
                        (x1, y1), (x2, y2),
                        (0, 255, 255), 3)
                    draw_target_info(annotated,
                        cx, cy, err_x, err_y,
                        rc_yaw, rc_throttle,
                        tid_int, cls_name)

                    # send to FC
                    if FC_CONNECTED and ser:
                        channels = [
                            RC_CENTER,    # roll  — pilot
                            RC_CENTER,    # pitch — pilot
                            rc_throttle,  # throttle — Jetson
                            rc_yaw,       # yaw — Jetson
                            RC_CENTER,
                            RC_CENTER,
                            RC_CENTER,
                            RC_CENTER,
                        ]
                        send_msp_rc(ser, channels)

                    writer.writerow([
                        time.time(), frame_count,
                        tid_int, cls_name,
                        x1, y1, x2, y2,
                        round(conf.item(), 3),
                        cx, cy, err_x, err_y,
                        rc_yaw, rc_throttle,
                        0, round(lat, 2)
                    ])

        # target lost — return to center, reset PID
        if target_lost:
            pid_yaw.reset()
            pid_throttle.reset()
            if FC_CONNECTED and ser:
                channels = [RC_CENTER] * 8
                send_msp_rc(ser, channels)
            writer.writerow([
                time.time(), frame_count,
                None, None,
                None, None, None, None, None,
                None, None, None, None,
                RC_CENTER, RC_CENTER,
                1, round(lat, 2)
            ])

        draw_status(annotated, FC_CONNECTED,
                    n_objects, fps, target_lost)
        out.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            print(
                f"Frame:{frame_count:5d} | "
                f"FPS:{fps:5.1f} | "
                f"Obj:{n_objects} | "
                f"Yaw:{rc_yaw} Thr:{rc_throttle} | "
                f"Lost:{target_lost} | "
                f"FC:{FC_CONNECTED}"
            )

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    cap.release()
    out.release()
    log.close()
    if ser:
        ser.close()
    print(f"Saved to {save_dir}")
