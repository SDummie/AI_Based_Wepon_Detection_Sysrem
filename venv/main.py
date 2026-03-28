"""
AI Weapon Detector — Upgraded
Features:
  • YOLOv8 real-time detection
  • Confidence threshold filter
  • Cooldown-based email alert WITH screenshot attached
  • Sound alert (Windows beep)
  • Auto screenshot saved to /detections/
  • Video clip recording triggered on detection
  • CSV detection log
  • On-screen HUD: FPS, counter, status bar
  • Color-coded boxes per weapon type
"""

import cv2
import time
import csv
import os
import smtplib
import winsound
import threading
from email.mime.text    import MIMEText
from email.mime.base    import MIMEBase
from email.mime.multipart import MIMEMultipart
from email              import encoders
from ultralytics        import YOLO

# ============================================================
#  CONFIGURATION
# ============================================================
EMAIL_SENDER   = "Arusrangra@gmail.com"
EMAIL_PASSWORD = "Arus@1234"       # Use Gmail App Password (16-char) — NOT your normal password!
EMAIL_RECEIVER = "Arusrangra@gmail.com"

CONFIDENCE_THRESHOLD = 0.20
ALERT_COOLDOWN_SEC   = 30
DEBUG_MODE           = True    # Shows ALL labels YOLO sees (yellow text, top-right)
WEAPON_LABELS        = {"knife", "gun"}

MODEL_PATH        = "yolov8n.pt"   # Falling back to standard YOLO model
DETECTIONS_FOLDER = "detections"
LOG_FILE          = os.path.join(DETECTIONS_FOLDER, "detection_log.csv")

# Colors per label (BGR)
LABEL_COLORS = {
    "knife"   : (0,   0,   220),
    "gun"     : (0,   0,   255),
    "pistol"  : (30,  30,  240),
    "rifle"   : (0,   80,  200),
    "sword"   : (100, 0,   200),
    "handgun" : (20,  20,  230),
}
DEFAULT_COLOR = (0, 0, 200)

RECORD_SECONDS = 5   # Seconds of video to save on detection
# ============================================================


# ---- Setup output folder ----
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)

# ---- CSV log ----
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "label", "confidence", "screenshot"])


def log_detection(label: str, conf: float, img_path: str) -> None:
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), label, f"{conf:.3f}", img_path
        ])


# ---- Sound alert (non-blocking) ----
def play_alert() -> None:
    def _beep():
        for _ in range(3):
            winsound.Beep(1200, 300)
            time.sleep(0.1)
    threading.Thread(target=_beep, daemon=True).start()


# ---- Email with screenshot ----
def send_email(label: str, conf: float, img_path: str) -> None:
    def _send():
        try:
            msg            = MIMEMultipart()
            msg["Subject"] = "⚠️ WEAPON DETECTED — Security Alert"
            msg["From"]    = EMAIL_SENDER
            msg["To"]      = EMAIL_RECEIVER

            body = (
                f"🚨 Weapon detected by AI surveillance system.\n\n"
                f"  • Weapon     : {label.upper()}\n"
                f"  • Confidence : {conf * 100:.1f}%\n"
                f"  • Time       : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "Screenshot is attached.\nPlease take immediate action."
            )
            msg.attach(MIMEText(body, "plain"))

            # Attach screenshot
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",
                                f'attachment; filename="{os.path.basename(img_path)}"')
                msg.attach(part)

            with smtplib.SMTP("smtp.gmail.com", 587) as srv:
                srv.starttls()
                srv.login(EMAIL_SENDER, EMAIL_PASSWORD)
                srv.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

            print(f"📧  Alert email sent  [{label}  {conf*100:.1f}%]")
        except Exception as e:
            print(f"❌  Email failed: {e}")

    threading.Thread(target=_send, daemon=True).start()


# ---- Video recorder helper ----
class ClipRecorder:
    def __init__(self, fps=20, size=(640, 480)):
        self._fps    = fps
        self._size   = size
        self._writer = None
        self._frames = 0
        self._max    = 0
        self._path   = ""

    def start(self, filename: str, total_frames: int):
        self._path   = filename
        self._max    = total_frames
        self._frames = 0
        fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(filename, fourcc, self._fps, self._size)
        print(f"🎥  Recording clip → {filename}")

    def write(self, frame):
        if self._writer and self._frames < self._max:
            self._writer.write(cv2.resize(frame, self._size))
            self._frames += 1
            if self._frames >= self._max:
                self._writer.release()
                self._writer = None
                print(f"🎬  Clip saved → {self._path}")

    @property
    def recording(self):
        return self._writer is not None


# ---- Drawing helpers ----
def draw_box(frame, x1, y1, x2, y2, label, conf):
    color = LABEL_COLORS.get(label, DEFAULT_COLOR)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Confidence bar (inside box, bottom edge)
    bar_w = x2 - x1
    fill  = int(bar_w * conf)
    cv2.rectangle(frame, (x1, y2 - 6), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(frame, (x1, y2 - 6), (x1 + fill, y2), color, -1)

    # Label chip
    text = f"{label.upper()}  {conf*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_hud(frame, fps, count_today, weapon_active, recording):
    h, w = frame.shape[:2]

    # Top-left info
    cv2.putText(frame, f"FPS: {fps:5.1f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
    cv2.putText(frame, f"Detections today: {count_today}",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
    if recording:
        cv2.putText(frame, "[REC]",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # Bottom status bar
    if weapon_active:
        bar_col  = (0, 0, 180)
        bar_text = "  [!] WEAPON DETECTED - ALERT SENT"
    else:
        bar_col  = (30, 30, 30)
        bar_text = "  [OK] MONITORING  |  No Threat Detected"

    cv2.rectangle(frame, (0, h - 38), (w, h), bar_col, -1)
    cv2.putText(frame, bar_text, (6, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    # Time top-right
    ts = time.strftime("%H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.putText(frame, ts, (w - tw - 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 255), 2)


# ============================================================
#  MAIN
# ============================================================
print("🔄  Loading YOLO model …")
model  = YOLO(MODEL_PATH)
print(f"✅  Model loaded: {MODEL_PATH}")

# --- Startup sound test so user knows audio is working ---
print("🔔  Playing startup beep (sound test) …")
try:
    winsound.Beep(1000, 400)
    print("✅  Sound OK")
except Exception as e:
    print(f"❌  Sound failed: {e}")

print("📹  Starting camera … (Press ESC to quit)")
print("👀  Watch the YELLOW text (top-right) — it shows everything YOLO sees.\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌  Cannot open camera.")
    exit()

cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam_fps = cap.get(cv2.CAP_PROP_FPS) or 20

recorder        = ClipRecorder(fps=cam_fps, size=(cam_w, cam_h))
last_alert_time = 0
detection_count = 0
prev_time       = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  Frame read error — exiting.")
        break

    now     = time.time()
    results = model(frame, verbose=False)

    weapon_detected = False
    best_label      = ""
    best_conf       = 0.0

    debug_labels = []   # collect all seen labels for overlay

    for r in results:
        for box in r.boxes:
            conf  = float(box.conf[0])
            label = model.names[int(box.cls[0])]

            # Debug: show every detection above 10% confidence
            if DEBUG_MODE and conf >= 0.10:
                debug_labels.append(f"{label} {conf*100:.0f}%")
                print(f"   👁  YOLO sees: {label:15s}  {conf*100:.1f}%")

            if label in WEAPON_LABELS and conf >= CONFIDENCE_THRESHOLD:
                weapon_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw_box(frame, x1, y1, x2, y2, label, conf)

                if conf > best_conf:
                    best_conf  = conf
                    best_label = label

    # Debug overlay — top-right corner, lists everything YOLO sees
    if DEBUG_MODE and debug_labels:
        for i, txt in enumerate(debug_labels[:10]):   # max 10 lines
            h_frame = frame.shape[0]
            w_frame = frame.shape[1]
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(frame, txt,
                        (w_frame - tw - 10, 55 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    # --- Trigger alert pipeline ---
    if weapon_detected and (now - last_alert_time) >= ALERT_COOLDOWN_SEC:
        detection_count += 1
        ts        = time.strftime("%Y%m%d_%H%M%S")
        img_path  = os.path.join(DETECTIONS_FOLDER, f"detect_{ts}.jpg")
        clip_path = os.path.join(DETECTIONS_FOLDER, f"clip_{ts}.mp4")

        cv2.imwrite(img_path, frame)
        log_detection(best_label, best_conf, img_path)
        play_alert()
        send_email(best_label, best_conf, img_path)

        if not recorder.recording:
            recorder.start(clip_path, int(cam_fps * RECORD_SECONDS))

        last_alert_time = now
        print(f"⚠️  [{best_label.upper()}]  conf={best_conf*100:.1f}%  img→{img_path}")

    # Feed recorder
    recorder.write(frame)

    # HUD
    fps       = 1.0 / (now - prev_time + 1e-9)
    prev_time = now
    draw_hud(frame, fps, detection_count, weapon_detected, recorder.recording)

    cv2.imshow("AI Weapon Detector", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n👋  Detector stopped.  Total detections this session: {detection_count}")
print(f"📁  Logs & screenshots saved in: ./{DETECTIONS_FOLDER}/")