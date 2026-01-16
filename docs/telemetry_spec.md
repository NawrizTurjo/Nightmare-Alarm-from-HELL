# Telemetry Specification

## Log Format

All logs are written to `logs/events.log` as newline-delimited JSON (NDJSON).

### Log Entry Schema

```json
{
  "ts": "2026-01-17T03:40:00.123Z",
  "fps": 15.2,
  "latency_ms": 45.3,
  "state": "SET_HOUR_TENS",
  "gesture": "FINGER_COUNT",
  "gesture_value": 2,
  "confidence": 0.92,
  "event": "HOLD_PROGRESS",
  "hold_progress": 0.75,
  "hand_detected": true,
  "num_hands": 1
}
```

## Metrics Captured

| Metric          | Type          | Description                        |
| --------------- | ------------- | ---------------------------------- |
| `ts`            | ISO timestamp | Event timestamp                    |
| `fps`           | float         | Current frames per second          |
| `latency_ms`    | float         | Frame processing time in ms        |
| `state`         | string        | Current FSM state                  |
| `gesture`       | string        | Detected gesture type              |
| `gesture_value` | int/null      | Gesture value (e.g., finger count) |
| `confidence`    | float         | Gesture detection confidence       |
| `event`         | string        | FSM event triggered                |
| `hold_progress` | float         | Progress of hold gesture (0-1)     |
| `hand_detected` | bool          | Whether hand is visible            |
| `num_hands`     | int           | Number of hands detected           |

## Event Types

- `FRAME_PROCESSED` - Regular frame processing
- `GESTURE_DETECTED` - New gesture recognized
- `HOLD_PROGRESS` - Hold gesture in progress
- `HOLD_COMPLETE` - Hold gesture completed
- `GLITCH_TRIGGERED` - Intentional glitch activated
- `STATE_CHANGE` - FSM state transition
- `ALARM_SCHEDULED` - Alarm set
- `ALARM_RING` - Alarm started ringing
- `ALARM_STOPPED` - Alarm stopped by user
- `ERROR` - Detection error

## Performance Targets

| Metric            | Target | Acceptable |
| ----------------- | ------ | ---------- |
| FPS               | 30     | ≥10        |
| Latency           | <100ms | <150ms     |
| Gesture accuracy  | 98%    | ≥95%       |
| False commit rate | <2%    | <5%        |

## Log Rotation

- **Max file size**: 10MB
- **Rotation**: Keep last 5 log files
- **Naming**: `events.log`, `events.log.1`, etc.
