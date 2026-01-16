# Finger Alarm Nightmare 🖐️⏰😱

> **The Worst Alarm Clock UI** — A gesture-only alarm clock for the Worst UI Competition

## What Is This?

An alarm clock that **hates you**. No buttons. No touchscreen. Just your webcam, your hands, and pure frustration.

- 🖐️ **Gesture-only control** — Set alarms by holding up fingers
- ⏱️ **2-second holds** — Every digit requires a 2-second hold
- 💥 **10% failure rate** — Inputs randomly fail at 90% progress
- 🎨 **Chaotic UI** — Colors change, text jitters, nothing makes sense

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open `http://localhost:8501` and allow camera access.

## How to Set an Alarm

| Digit | Gesture                   |
| ----- | ------------------------- |
| 0     | Fist (no fingers)         |
| 1-5   | Hold up that many fingers |
| 6     | 5 fingers + swipe UP      |
| 7     | 5 fingers + swipe RIGHT   |
| 8     | 5 fingers + swipe LEFT    |
| 9     | 5 fingers + swipe DOWN    |

**Hold each gesture for 2 seconds** until the loading bar fills.

To **confirm**: Thumbs up 👍  
To **stop alarm**: Two open hands 🙌

## Tech Stack

- **Streamlit** + **streamlit-webrtc** — Real-time web video
- **MediaPipe Hands** — Hand landmark detection
- **OpenCV** — Frame processing and UI overlays
- **APScheduler** — Alarm scheduling

## Docker

```bash
docker build -t finger-alarm-nightmare .
docker run -p 8501:8501 finger-alarm-nightmare
```

## Documentation

- [FSM Specification](docs/fsm_spec.md) — State machine design
- [Gesture Specification](docs/gesture_spec.md) — Gesture vocabulary
- [UI Specification](docs/ui_spec.md) — Visual overlay details
- [Safety Note](docs/safety_note.md) — Accessibility disclaimer
- [Deployment Guide](docs/deploy.md) — Setup instructions

## ⚠️ Disclaimer

This is **intentionally terrible**. It's a satirical demonstration of hostile design for a competition. Don't use this as your actual alarm clock.

---

_Built with ❤️ and spite for the Worst UI Competition_
