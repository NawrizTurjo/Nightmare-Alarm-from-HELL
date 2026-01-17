# Nightmare Alarm from HELL 🖐️⏰😱

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nightmare-alarm-from-hell.streamlit.app/)

> **The Worst Alarm Clock UI** — A gesture-only alarm clock for the Worst UI Competition.

## 🔥 What Is This?

An alarm clock that **hates you**. No buttons. No touchscreen. Just your webcam, generic computer vision, and pure frustration.

- 🖐️ **Gesture-only control** — Set alarms by holding up fingers (good luck).
- ⏱️ **2-second holds** — Every digit requires a painful 2-second hold.
- 💥 **10% failure rate** — Inputs randomly fail at 90% progress (it's a feature).
- 🎨 **Chaotic UI** — Colors scream, text jitters, nothing makes sense.
- 🚫 **No Stop Button** — To stop the alarm, you must perform a random "Challenge" (e.g., "Show 3 fingers with left hand and 2 with right").

## 🚀 Quick Start

### Local Run

```bash
# Mobile/Headless users: This needs a webcam!
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` and allow camera access.

## ☁️ Deployment (Streamlit Cloud)

This app is optimized for **Streamlit Community Cloud**.
If you fork this, note the following critical files for cloud compatibility:

- **`packages.txt`**: Installs system dependencies (`libgl1`) for OpenCV.
- **`.python-version`**: Pins Python to `3.11` (required for MediaPipe/Audio compatibility).
- **`requirements.txt`**: Uses `opencv-python-headless` to avoid server crashes.

### 🌐 TURN Server (Crucial for Cloud!)

If the video stays on "Loading..." or errors out, you need a **TURN server** (to punch through firewalls).

1.  Create a free account at [Metered.ca](https://www.metered.ca/).
2.  Go to your Streamlit Cloud Dashboard → **Settings** → **Secrets**.
3.  Add this config:

```toml
[ice_servers]
urls = ["turn:global.relay.metered.ca:80"]
username = "YOUR_METERED_USERNAME"
credential = "YOUR_METERED_PASSWORD"
```

The app will automatically use these credentials!

## 📖 How to Suffer (User Guide)

| Digit   | Gesture                                    |
| ------- | ------------------------------------------ |
| **0**   | ✊ Fist (no fingers)                       |
| **1-5** | 🖐️ Hold up that many fingers               |
| **6-9** | 🖐️ **Hold 5 fingers > 3s** (Roulette Mode) |

1.  **Hold each gesture** until the loading bar fills.
2.  **Confirm**: Thumbs up 👍 (Hold 0.8s).
3.  **Stop Alarm**: **Read the screen!** It will demand a specific gesture (e.g., "Double High Five").

## 🛠️ Tech Stack

- **Streamlit** + **WebRTC** — Real-time video processing in Python.
- **MediaPipe Hands** — fast hand tracking.
- **OpenCV** — Drawing the terrible UI on the video frames.
- **PyDub** — For generating annoying audio.

## ⚠️ Disclaimer

This is **intentionally terrible**. It is a satirical demonstration of hostile design.
**DO NOT** rely on this for important alarms. You have been warned.

---

_Built with ❤️ and spite for the Worst UI Competition_
