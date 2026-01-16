# UI Specification

## Overlay Layout (burned into frames)

```
┌────────────────────────────────────────────────────────────────┐
│ STATE: SET_HOUR_TENS              (top-left, red, scale 1.0)  │
│ ══════════════════                                             │
│ 0_:__                             (below state, blue, 2.2)    │
│                                                                │
│                    👆                                          │
│               (ghost gesture,                                  │
│                small, off-center)                              │
│                                                                │
│                                                                │
│                                                                │
│ ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░              ┌─────────┐                 │
│ [Progress Bar]                     │ GLITCH! │ (bottom-right)  │
│ (bottom-left)                      └─────────┘                 │
└────────────────────────────────────────────────────────────────┘
```

## Element Specifications

### State Label (Top-Left)

- **Position**: `(20, 40)`
- **Font**: `cv2.FONT_HERSHEY_SIMPLEX`
- **Scale**: 1.0
- **Color**: Red `(0, 0, 255)` BGR
- **Thickness**: 2

### Current Time Display

- **Position**: `(20, 100)`
- **Scale**: 2.2
- **Color**: Blue `(255, 0, 0)` BGR
- **Thickness**: 3
- **Format**: `HH:MM` with underscores for unset digits

### Progress Loading Bar

- **Position**: `(50, frame_h - 120)` to `(350, frame_h - 90)`
- **Background**: Dark gray `(50, 50, 50)`
- **Fill**: Red `(0, 0, 255)`
- **Fill width**: Scales linearly with hold progress (0-2.0s)

### Glitch Overlay (Bottom-Right)

- **Position**: `(frame_w - 150, frame_h - 80)`
- **Size**: 140x50 pixels
- **Color**: Flashing red at 1 Hz
- **Text**: "GLITCH!" in white

### Ghost Gesture Icon (Center)

- **Position**: Center-offset by `(-50, 30)`
- **Opacity**: 30% translucent
- **Content**: Icon showing required gesture
- **Size**: Small (for Worst UI effect)

## Worst UI Visual Effects

### Text Jitter

- **Interval**: Every 600ms
- **Range**: ±5 pixels X and Y
- **Applied to**: All text elements

### Color Palette Chaos

- **Interval**: Random 7-12 seconds
- **Effect**: All colors randomly change
- **Palettes**: Clashing neon colors preferred

### Text Rotation

- **Range**: ±2-4 degrees
- **Applied to**: Time display text
- **Timing**: Random intervals

### Label Swap

- **Effect**: "LEFT" and "RIGHT" labels swap randomly
- **Purpose**: Maximum confusion

### Fake Success Popup

- **Text**: "Alarm Set — (Actually not)"
- **Duration**: 2 seconds
- **Frequency**: Random, occasional

## Alarm Ringing Visuals

- **Background**: Flashing red/white at 2 Hz
- **Text**: "WAKE UP!!!" large, centered
- **Shake effect**: Frame offset ±10 pixels randomly

## Error State Overlay

- **Text**: "HAND NOT DETECTED"
- **Subtext**: "WINK TO CONTINUE" (if blink detection enabled)
- **Background**: Semi-transparent dark overlay
- **Color**: Yellow warning `(0, 255, 255)`
