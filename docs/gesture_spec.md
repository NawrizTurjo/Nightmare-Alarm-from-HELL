# Gesture Specification

## Primary Gesture Vocabulary

### Finger Count Detection

| Fingers       | Digit | Gesture Description            |
| ------------- | ----- | ------------------------------ |
| 0 (fist)      | 0     | All fingers folded             |
| 1             | 1     | Index finger extended          |
| 2             | 2     | Index + middle extended        |
| 3             | 3     | Index + middle + ring extended |
| 4             | 4     | All except thumb extended      |
| 5 (open palm) | 5     | All fingers extended           |

### Modifier Gestures (for digits 6-9)

| Base      | Modifier    | Result Digit |
| --------- | ----------- | ------------ |
| 5 fingers | Swipe UP    | 6            |
| 5 fingers | Swipe RIGHT | 7            |
| 5 fingers | Swipe LEFT  | 8            |
| 5 fingers | Swipe DOWN  | 9            |

**Modifier Window**: 1.2 seconds after open palm detection

### Control Gestures

| Gesture        | Hold Duration | Action                        |
| -------------- | ------------- | ----------------------------- |
| THUMBS_UP      | 0.8s          | Confirm (in CONFIRM state)    |
| FIST           | 1.2s          | Clear current field           |
| TWO_HANDS_OPEN | 0.8s          | Stop alarm (in ALARM_RINGING) |

## Detection Heuristics

### Finger Count Algorithm

1. Get MediaPipe landmarks for tips: `[4, 8, 12, 16, 20]`
2. **Thumb detection** (handedness-aware):
   - Right hand: extended if `tip[4].x < ip[3].x`
   - Left hand: extended if `tip[4].x > ip[3].x`
3. **Other fingers**: extended if `tip.y < pip.y` (tip above PIP joint)
4. **Persistence**: Require same state for 3 consecutive frames
5. **Smoothing**: EWMA with α = 0.6

### Swipe Detection

- Track hand centroid over 6 frames
- **Threshold**: `SWIPE_MIN_DIST_PIX = 70` pixels
- **Direction**: Angle from horizontal ±45°
  - RIGHT: -45° to +45°
  - UP: +45° to +135°
  - LEFT: +135° to -135°
  - DOWN: -135° to -45°

### Thumbs-Up Detection

Requirements:

- Thumb tip above index MCP
- Other 4 fingers folded
- Hold for 0.8 seconds
- Confidence threshold: 0.75

### Two Hands Open Detection

Requirements:

- Two hands detected
- Both palms open ratio ≥ 0.8
- Hold for 0.8 seconds

## Thresholds & Constants

| Constant           | Value | Description                     |
| ------------------ | ----- | ------------------------------- |
| HOLD_SEC           | 2.0s  | Hold time for digit commit      |
| HOLD_CONFIRM_SEC   | 0.8s  | Hold time for thumbs-up/stop    |
| CONF_THRESH        | 0.85  | Gesture confidence threshold    |
| EWMA_ALPHA         | 0.6   | Smoothing factor                |
| PERSIST_FRAMES     | 3     | Frames for stability            |
| SWIPE_MIN_DIST_PIX | 70    | Minimum swipe distance          |
| MODIFIER_WINDOW    | 1.2s  | Time to detect swipe after palm |

## Glitch Behavior

- **Probability**: 10% on commit attempts
- **Trigger**: At 90% progress
- **Effect**: Reset progress to 0, show "GLITCH" overlay
- **Purpose**: Intentional frustration (Worst UI)
