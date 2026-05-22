# 🕶️ VisionAid — Smart Wearable Device for the Visually Impaired

> A wearable assistive device that delivers real-time spoken navigation alerts using AI object detection — built on a Raspberry Pi 3b+.

---

## 📌 Overview

Over 250 million people worldwide live with visual impairment and struggle with everyday navigation — identifying obstacles, estimating distances, or recognizing objects around them. Commercial solutions exist but are often expensive and inaccessible.

**VisionAid** is a wearable prototype that combines Yolo object detection, Rasberry Pi, Pi camera and audio feedback to give visually impaired users real-time awareness of their surroundings — built entirely with off-the-shelf hardware and open-source AI, for under $90.

### What it does
- 📷 Continuously captures video via a small camera mounted on a glasses frame
- 🤖 Detects objects, people, and obstacles in real time using **YOLOv8n**
- 🔊 Delivers spoken audio alerts through a **bluetooth earpiece**
- ⚡ Prioritizes safety-critical alerts (close obstacles) over general scene descriptions

---

## 🏗️ System Architecture

```
PiCamera → Raspberry Pi 3b+ → YOLOv8n Detection (ncnn format) → Audio Output (TTS / Pygame)
```
---

## 🛠️ Hardware Components

| Component | Model / Spec | Purpose | Est. Cost |
|-----------|-------------|---------|-----------|
| Processing Unit | Raspberry Pi 3b+ | Runs all code and AI model |
| Camera | Raspberry Pi Camera Module 3 | Captures live video feed |
| Audio Output | Earpiece (Bluetooth) | Speaks alerts to the user |
| Power | USB-C Power Bank (5000mAh+) | Portable power supply |
| Frame | Hat + mounting tape | Wearable mounting |

> **Why bone conduction?** Standard earphones block ambient sound — dangerous for visually impaired users who need to hear traffic and voices. Bone conduction transmits audio through the cheekbones directly to the inner ear, leaving the ear canal open.

---

## 💻 Software Stack

| Layer | Tool / Library | Why |
|-------|---------------|-----|
| Language | Python 3.10+ | Easy to use, great Raspberry Pi support |
| Object Detection | YOLOv8n (Ultralytics) | Lightweight, runs offline on Pi, 80+ object classes |
| Camera Interface | OpenCV (`cv2`) | Industry-standard video capture and frame processing |
| Text-to-Speech | `gTTS` | low latency, but need internet connection |
| Audio Playback | `pygame.mixer` | Pre-recorded audio clips for faster response time |

---

## 🚀 Getting Started

### Prerequisites
- Raspberry Pi 3b+ running Raspberry Pi OS (64-bit recommended)
- Python 3.10+
- Raspberry Pi Camera Module connected and enabled via `raspi-config`
- Bluetooth earpiece paired


### Run

```bash
python main.py
```

> On first run, YOLOv8n weights (~6MB) will be downloaded automatically via the `ultralytics` package.

---



---

## ⚠️ Known Challenges

| Challenge | Risk | Mitigation |
|-----------|------|-----------|
| Latency | High | Use YOLOv8 nano model, reduce frame rate, prioritize ultrasonic for instant alerts |
| Battery Life | Medium | Test early, reduce CPU load, consider Pi Zero 2W for future iteration |
| False Positives | Medium | Set minimum confidence threshold (≥60%), ignore detections below it |
| Audio Overload | Medium | 5-second cooldown per object class; only announce new/changed objects |
| Physical Mounting | Low | Adjustable clips, lightweight power bank, careful cable management |
| Lighting Conditions | Medium | Test in varied lighting; consider adding IR LED for low-light environments |

---

## 🔮 Future Improvements

- **Text Reading (OCR)** — Use Tesseract OCR to read signs, labels, and text aloud
- **Face Recognition** — Identify known people and announce their name
- **GPS Integration** — Add a GPS module for turn-by-turn navigation directions
- **Cloud AI Mode** — Send frames to a Vision API for richer scene descriptions
- **Lighter Hardware** — Migrate to Raspberry Pi Zero 2W + custom 3D-printed frame
- **Mobile Companion App** — Let caregivers configure the device and review alerts remotely
- **Haptic Feedback** — Vibration motor in the frame for silent distance alerts (closer = faster vibration)

---

---

> *Built with open-source tools and a clear goal — to make independent navigation accessible to everyone.*
