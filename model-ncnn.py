"""
VisionAid — Real-time object detection + spatial audio for blind users
Model  : YOLOv8n NCNN via ncnn Python bindings (no torch, no ultralytics)
Camera : Raspberry Pi Camera (picamera2)

FOLDER STRUCTURE:
    final-1.py
    yolov8n_ncnn_model/
        model.ncnn.param
        model.ncnn.bin
    dnn_model/
        classes.txt
    audio_cache/          ← auto-created
"""

import cv2
import numpy as np
import os
import time
import threading
import pygame
import ncnn
# from picamera2 import Picamera2

# ═══════════════════════════════════════════════════════════════
# CONFIG  — tweak these without touching anything else
# ═══════════════════════════════════════════════════════════════
NCNN_PARAM       = "yolov8n_ncnn_model/model.ncnn.param"
NCNN_BIN         = "yolov8n_ncnn_model/model.ncnn.bin"
CLASSES_FILE     = "dnn_model/classes.txt"

CONFIDENCE_THRESHOLD = 0.60    # detections below this are ignored
NMS_THRESHOLD        = 0.45    # overlap threshold for non-max suppression
INPUT_W              = 640     # model input width
INPUT_H              = 640     # model input height
GLOBAL_COOLDOWN_SEC  = 3.0     # minimum seconds between ANY two announcements
PER_OBJECT_COOLDOWN  = 4.0     # minimum seconds before repeating the SAME object
CAMERA_W             = 640     # camera capture width
CAMERA_H             = 480     # camera capture height
TTS_SLOW             = False   # True = slower clearer speech on cheap speakers

# ── Priority tiers ──────────────────────────────────────────────
HIGH_PRIORITY = {
    "person", "car", "truck", "bus", "bicycle", "motorbike",
    "traffic light", "stop sign", "fire hydrant", "dog", "cat"
}
MEDIUM_PRIORITY = {
    "chair", "dining table", "bed", "toilet",
    "bottle", "cup", "knife", "scissors", "cell phone", "laptop"
}
PRIORITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio_cache")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def get_priority(class_name: str) -> str:
    if class_name in HIGH_PRIORITY:
        return "HIGH"
    if class_name in MEDIUM_PRIORITY:
        return "MEDIUM"
    return "LOW"


def get_position_label(x: int, w: int, frame_width: int) -> str:
    """Map bounding-box centre-x to a spoken position."""
    cx    = x + w / 2
    third = frame_width / 3
    if cx < third:
        return "on your left"
    elif cx < 2 * third:
        return "ahead"
    else:
        return "to your right"


# ═══════════════════════════════════════════════════════════════
# AUDIO ENGINE  (non-blocking — runs in background thread)
# ═══════════════════════════════════════════════════════════════
pygame.mixer.init()
_audio_lock = threading.Lock()
_is_playing = False


def _play_worker(filepath: str):
    global _is_playing
    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
    with _audio_lock:
        _is_playing = False


def speak(text: str):
    """Generate TTS (cached to disk) and play without blocking the main loop."""
    global _is_playing
    from gtts import gTTS

    safe_name = text.replace(" ", "_")[:80] + ".mp3"
    filepath  = os.path.join(AUDIO_DIR, safe_name)

    if not os.path.isfile(filepath):
        tts = gTTS(text=text, lang="en", slow=TTS_SLOW)
        tts.save(filepath)

    with _audio_lock:
        _is_playing = True

    threading.Thread(target=_play_worker, args=(filepath,), daemon=True).start()


def is_audio_busy() -> bool:
    with _audio_lock:
        return _is_playing


# ═══════════════════════════════════════════════════════════════
# LOAD CLASSES
# ═══════════════════════════════════════════════════════════════
classes = {}
with open(os.path.join(BASE_DIR, CLASSES_FILE), "r") as f:
    for i, line in enumerate(f):
        classes[i] = line.strip()
print(f"[VisionAid] Loaded {len(classes)} classes.")


# ═══════════════════════════════════════════════════════════════
# LOAD NCNN MODEL
# ═══════════════════════════════════════════════════════════════
net = ncnn.Net()
net.opt.use_vulkan_compute = False   # CPU only — safe for Pi
net.load_param(os.path.join(BASE_DIR, NCNN_PARAM))
net.load_model(os.path.join(BASE_DIR, NCNN_BIN))
print("[VisionAid] NCNN model loaded.")


# ═══════════════════════════════════════════════════════════════
# DETECTION FUNCTION
# ═══════════════════════════════════════════════════════════════
def detect_ncnn(frame: np.ndarray) -> list:
    """
    Run YOLOv8n-NCNN inference on a BGR frame.
    Returns list of (class_id, score, (x, y, w, h)) in pixel coords.
    """
    img_h, img_w = frame.shape[:2]

    # Preprocess — resize + normalize
    mat_in = ncnn.Mat.from_pixels_resize(
        frame,
        ncnn.Mat.PixelType.PIXEL_BGR2RGB,
        img_w, img_h,
        INPUT_W, INPUT_H
    )
    mean_vals = [0.0, 0.0, 0.0]
    norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)

    # Inference
    ex = net.create_extractor()
    ex.input("in0", mat_in)
    _, mat_out = ex.extract("out0")

    # Parse output — YOLOv8 NCNN layout: [x_c, y_c, w, h, cls_scores...]
    num_proposals = mat_out.h
    num_values    = mat_out.w

    boxes, scores_list, class_ids = [], [], []

    # Replace the entire parsing loop with this:
    output = np.array(mat_out).reshape(mat_out.h, mat_out.w)  # shape: (84, 8400)
    output = output.T  # transpose to (8400, 84)

    for i in range(output.shape[0]):
        cls_scores = output[i, 4:]
        class_id   = int(np.argmax(cls_scores))
        score      = float(cls_scores[class_id])

        if score < CONFIDENCE_THRESHOLD:
            continue

        cx = output[i, 0] / INPUT_W * img_w
        cy = output[i, 1] / INPUT_H * img_h
        bw = output[i, 2] / INPUT_W * img_w
        bh = output[i, 3] / INPUT_H * img_h

        x = int(cx - bw / 2)
        y = int(cy - bh / 2)
        w = int(bw)
        h = int(bh)

        boxes.append([x, y, w, h])
        scores_list.append(score)
        class_ids.append(class_id)

    # Non-max suppression
    detections = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(
            boxes, scores_list, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
        )
        for idx in indices:
            i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
            detections.append((class_ids[i], scores_list[i], tuple(boxes[i])))

    return detections


# ═══════════════════════════════════════════════════════════════
# CAMERA SETUP (Pi Camera via picamera2)
# ═══════════════════════════════════════════════════════════════
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)

def read_frame():
    """Read one frame from the laptop webcam."""
    ret, frame = cam.read()
    return frame if ret else None



# ═══════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════
last_announced_time: dict[str, float] = {}
last_global_announce: float           = 0.0

# ═══════════════════════════════════════════════════════════════
# MAIN LOOP  (headless — no display window)
# ═══════════════════════════════════════════════════════════════
print("[VisionAid] Running headless — press Ctrl+C to quit.")

try:
    while True:
        frame = read_frame()
        if frame is None:
            continue

        frame_h, frame_w = frame.shape[:2]
        now              = time.time()

        # ── Run detection ─────────────────────────────────────────
        detections = detect_ncnn(frame)

        # ── Build candidate list ──────────────────────────────────
        candidates = []

        for class_id, score, bbox in detections:
            class_name = classes.get(class_id, "unknown")
            x, y, w, h = bbox
            area        = w * h

            priority    = get_priority(class_name)
            pos_label   = get_position_label(x, w, frame_w)
            cooldown_ok = (now - last_announced_time.get(class_name, 0)) > PER_OBJECT_COOLDOWN

            if cooldown_ok:
                candidates.append(
                    (PRIORITY_RANK[priority], area, class_name, pos_label, score)
                )

        # ── Pick ONE winner — highest priority, then largest (= closest) ──
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

        global_ok = (now - last_global_announce) > GLOBAL_COOLDOWN_SEC

        if candidates and global_ok and not is_audio_busy():
            _, _, best_class, best_pos, best_score = candidates[0]
            phrase = f"{best_class} {best_pos}"
            print(f"[ANNOUNCE] {phrase}  (conf={best_score:.2f})")
            speak(phrase)
            last_announced_time[best_class] = now
            last_global_announce            = now
        else:
            # Print a heartbeat so you know the loop is alive
            obj_count = len(detections)
            if obj_count:
                names = [classes.get(d[0], "?") for d in detections]
                print(f"[DETECT]   {obj_count} object(s): {', '.join(names)}")

except KeyboardInterrupt:
    print("\n[VisionAid] Interrupted by user.")

finally:
    picam2.stop()
    pygame.mixer.quit()
    print("[VisionAid] Stopped.")
