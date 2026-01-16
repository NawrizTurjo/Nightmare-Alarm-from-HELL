# Deployment Guide

## Local Development

### Prerequisites

- Python 3.11+
- Webcam
- Modern browser (Chrome/Firefox recommended)

### Setup

```bash
# Clone repository
cd Finger-Alarm-Nightmare

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Access

Open `http://localhost:8501` in your browser.

## Docker Deployment

### Build

```bash
docker build -t finger-alarm-nightmare .
```

### Run

```bash
docker run -p 8501:8501 finger-alarm-nightmare
```

### With Dev Mode

```bash
docker run -p 8501:8501 -e DEV_MODE=true finger-alarm-nightmare
```

## Demo Checklist

### Before Demo

- [ ] Test webcam works
- [ ] Good lighting in room
- [ ] Close other camera apps
- [ ] Test audio output
- [ ] Enable `DEV_MODE` for emergency stop

### During Demo

- [ ] Allow camera permission when prompted
- [ ] Allow audio permission if asked
- [ ] Keep hand 1-2 feet from camera
- [ ] Use clear, deliberate gestures

### Troubleshooting

| Issue                   | Solution                      |
| ----------------------- | ----------------------------- |
| Camera not detected     | Check browser permissions     |
| Low FPS                 | Close other applications      |
| Gestures not recognized | Improve lighting, move closer |
| Audio not playing       | Click anywhere on page first  |
