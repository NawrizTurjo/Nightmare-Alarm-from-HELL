# 1 — Project Summary & Constraints

**Goal:** A Streamlit app (web) running `streamlit-webrtc` that allows users to set and stop alarms **using only hand gestures** (no on-screen buttons). UI elements appear burned into the video frames. The UX should be intentionally painful (Worst UI competition) but technically robust and demonstrable.

**Hard constraints**

* No on-screen/touch buttons for primary flows (setting, confirming, stopping alarms).
* Gestures only via webcam input.
* Use MediaPipe Hands (or equivalent) for landmarks + `streamlit-webrtc` for real-time video.
* App must be reliably demoable on a typical laptop (no GPU required).
* Antigravity agents will produce the code, tests, docs, and CI artifacts.

# 2 — Deliverables (what your agents must output)

For each item, specify file name(s) and a short description.

1. `app.py` — Streamlit entrypoint. Injects cursed CSS, launches `webrtc_streamer`, and displays small side info text.
2. `video_processor.py` — Contains `AlarmProcessor` (class blueprint) that the WebRTC thread uses. Includes all FSM logic, gesture detection hooks, and frame drawing utilities.
3. `fsm_spec.md` — Human-readable FSM diagram and transition table (states, events, guards, actions).
4. `gesture_spec.md` — Exact gesture vocabulary, mapping to actions, threshold numbers, and detection heuristics.
5. `ui_spec.md` — Exact on-screen visuals burned into frames (positions, font sizes, colors, loading-bar coordinates).
6. `requirements.txt`
7. `Dockerfile` — Container for running the app.
8. `tests/` — Unit & integration tests:

   * `test_fsm.py` (state transitions)
   * `test_gestures.py` (gesture classification simulations)
   * `test_video_pipeline.py` (frame pipeline sanity)
9. `README.md` — Setup, run, demo script, judge talking points.
10. `deploy.md` — Deployment & demo instructions (local + container).
11. `ci.yml` — GitHub Actions skeleton to run tests and build container.
12. `presentation_script.md` — 2–3 minute demo narration + 1-min judges defense.
13. `telemetry_spec.md` — Logging and metrics to capture (FPS, latency, gesture_confidence).
14. `safety_note.md` — Accessibility & safety disclaimers (explicitly describing intent and fallback controls for dev mode).

# 3 — High-Level Architecture (components & interactions)

1. **Streamlit UI (Main Thread)**

   * Ugly CSS injection and static text. Minimal widgets (only for debug/dev mode; hidden in production).
   * Starts `webrtc_streamer(video_processor_factory=AlarmProcessor)`.

2. **WebRTC Thread — `AlarmProcessor`**

   * Receives frames (`recv`) from webrtc.
   * Runs MediaPipe Hands on each frame (flip horizontally).
   * Computes: landmarks, finger count (0–5), swipe vectors, thumb/thumbs-up detection, two-hand detection, face-blink (optional) via MediaPipe Face Mesh.
   * Manages FSM, gesture hold timers, glitch injection, commit and rejection logic.
   * Draws all UI overlays on returned frames (`cv2.putText`, `rectangle`).

3. **Alarm Manager (inside Processor)**

   * Maintains scheduled alarms (in-memory).
   * Alarm scheduler uses `threading.Timer` or `apscheduler` inside processor or off-thread manager for ringing.
   * When alarm fires, audio is played (via browser: `webrtc` or via stream the app can trigger an audio file to play in the page using a small JS shim or WebAudio fallback) — accept agent to pick the most reliable approach for streamlit-webrtc environment.

4. **Persistence (Optional)**

   * Local JSON file for alarms persistence between runs: `alarms.json`.
   * Format: timestamp, label, set_by_sequence, created_at.

5. **Logging / Telemetry**

   * Write logs (structured JSON) to `logs/` with `level, timestamp, fps, gesture_confidence, state`.
   * Capture metrics: FPS (avg), processing_latency(ms), false_commit_rate.

# 4 — Full Finite State Machine (FSM)

Provide this whole table to agents.

**States (explicit names):**

* `IDLE` — no active flow, shows default overlay
* `SET_HOUR_TENS`
* `SET_HOUR_ONES`
* `SET_MIN_TENS`
* `SET_MIN_ONES`
* `CONFIRM` — final confirmation (thumbs-up)
* `ALARM_SET` — alarm scheduled
* `ALARM_RINGING`
* `ACK_RING` — stop/acknowledge flow
* `ERROR` — used for detection failures

**Events (inputs):**

* `FINGER_COUNT(n)` where n in 0..5
* `HOLD_COMPLETE` — user held same gesture for `HOLD_SEC`
* `SWIPE(dir)` — dir in `LEFT, RIGHT, UP, DOWN`
* `THUMBS_UP`
* `TWO_HANDS_OPEN`
* `FIST` (clear)
* `HAND_LOST` (no hands for `HAND_LOST_TIMEOUT`)
* `FACE_BLINK` (optional sabotage/cancel)
* `TIME_TIMEOUT` — general timeout for waiting in state

**Guards & Actions**

* Guards include `valid_digit_for_field(digit, field)`, `confidence_above(threshold)`, `not_glitched()`
* Actions: `commit_digit(digit)`, `advance_state()`, `emit_feedback(success|fail)`, `schedule_alarm(time)`, `start_ring()`, `stop_ring()`, `play_sound()`, `persist_alarm()`

**Initial State:** `SET_HOUR_TENS` (or `IDLE` then go to set on gesture)

**Transitions Example (concise):**

* `SET_HOUR_TENS` + `HOLD_COMPLETE` & `digit in 0..2` → commit hour_tens = digit → `SET_HOUR_ONES`
* `SET_HOUR_ONES` + `HOLD_COMPLETE` & `digit in allowed_range(hour_tens)` → commit → `SET_MIN_TENS`
* `SET_MIN_TENS` + `HOLD_COMPLETE` & `digit in 0..5` → commit → `SET_MIN_ONES`
* `SET_MIN_ONES` + `HOLD_COMPLETE` & `digit in 0..9`(via mapping) → commit → `CONFIRM`
* `CONFIRM` + `THUMBS_UP` & confidence → `ALARM_SET` (call schedule)
* `ALARM_RINGING` + `TWO_HANDS_OPEN` → `ACK_RING` → `ALARM_STOPPED` (persist stop)
* Any state + `HAND_LOST` for > `HAND_LOST_TIMEOUT` → visual `ERROR` overlay + return to last stable state after `RETRY_TIMEOUT`

# 5 — Gesture Vocabulary & Exact Mapping (do not deviate)

Agents need these exact rules:

## Primary inputs (detectors)

* `FINGER_COUNT` — integer 0..5 from MediaPipe logic (see CV heuristics below).
* `SWIPE` — computed via centroid delta per frame over last `N=4` frames. Vector threshold `SWIPE_MIN_DIST_PIX = 70` and direction determinable by angle.
* `THUMBS_UP` — identify thumb tip above index MCP and thumb extended; require `thumb_up_confidence >= 0.75`.
* `TWO_HANDS_OPEN` — two hands detected and both palm-open ratio ≥ 0.8.
* `FIST` — all fingertips folded.

## Core mapping (exact)

* `0 fingers` (fist) => digit 0
* `1 finger` => 1
* `2 fingers` => 2
* `3 fingers` => 3
* `4 fingers` => 4
* `5 fingers` (open palm) => base-5 digit modifier (see below for mapping 6–9)

### Mapping 6–9 (choose **one** of two strategies; the plan forces exactly one — pick Modifier Gesture method for clarity)

**Modifier Gesture (recommended):**

* `5 fingers` + `SWIPE_UP` = 6
* `5 fingers` + `SWIPE_RIGHT` = 7
* `5 fingers` + `SWIPE_LEFT` = 8
* `5 fingers` + `SWIPE_DOWN` = 9

(Agent must implement logic: when open palm is detected, read next motion vector within `MODIFIER_WINDOW = 1.2s` to interpret as mapping.)

### Confirm & Cancel gestures

* `THUMBS_UP` (hold 0.8s) → Confirm (only in `CONFIRM` state)
* `FIST` (hold 1.2s) → Clear current field
* `TWO_HANDS_OPEN` (hold 0.8s) while `ALARM_RINGING` -> Stop alarm
* `HAND_DISAPPEAR` for `HAND_LOST_TIMEOUT = 5s` → cancel or go to `ERROR`

### Holding rules

* `HOLD_SEC = 2.0` seconds required to trigger `HOLD_COMPLETE` for digit commits.
* Visual loading bar must fill linearly during hold (0–2.0s).

### Confidence & rejection

* Use an exponential moving average filter on finger count to avoid jitter: `alpha = 0.6`.
* Gesture confidence threshold for committing: `CONF_THRESH = 0.85`. If measured (e.g., via landmark stability or MediaPipe score) is below this, show “gesture unclear” and require another hold attempt.
* Agents implement `glitch`: on commit attempt with probability `GLITCH_P = 0.10` reject at 90% progress and reset progress to 0, show `GLITCH` overlay.

# 6 — CV / ML Detection Heuristics (practical and exact)

Agents must implement the following precise, testable heuristics. They are not learning tasks — they are rules.

## MediaPipe config

* `max_num_hands = 1` for gesture flows (except when checking for two hands).
* `min_detection_confidence = 0.7`
* `min_tracking_confidence = 0.6`

## Finger Count algorithm (exact steps to implement)

1. From MediaPipe hand landmarks, use tip indices: `[4,8,12,16,20]`.
2. Thumb detection: compare tip.x to IP or MCP.x depending on handedness:

   * If right hand (handedness from mediapipe): thumb extended if `tip[4].x < ip[3].x` (mirror as needed for flipped frame).
3. For each non-thumb finger, finger is extended if `tip.y < pip.y` (tip above pip in image coordinate).
4. For robustness, require same finger states persist for `PERSIST_FRAMES = 3` frames before changing current count.
5. Compute `finger_count = sum(extended_flags)` and apply EWMA smoothing: `s = alpha*new + (1-alpha)*s`.

## Swipe detection

* Track hand centroid across `N = 6` frames.
* If X delta > `SWIPE_MIN_DIST_PIX` and average speed > `SWIPE_MIN_VEL` => `SWIPE_RIGHT` (similar for left/up/down using Y).
* Angle thresholds: ±45 degrees tolerance.

## Thumbs-up detection

* Thumb tip above index MCP AND other fingers folded for >= `HOLD_SEC_CONFIRM = 0.8s`.
* Use normalized landmark distances to compute confidences and average for `thumbs_confidence`.

## Face-blink detection (optional sabotage)

* Use MediaPipe Face Mesh. Eye Aspect Ratio (EAR) threshold = 0.22 for blink detection.

# 7 — UI overlays & visuals (exact pixels / guidelines for consistency)

Positions should be specified relative to frame size; agents should implement helper functions.

* **Top-left**: `STATE` label. Font: `FONT_HERSHEY_SIMPLEX`, scale 1.0, color red (BGR (0,0,255)), thickness 2.
* **Below STATE**: `CURRENT SET` time text, big: scale 2.2, color blue (255,0,0), thickness 3.
* **Bottom-left**: Loading bar: rectangle from `(50, frame_h - 120)` to `(350, frame_h - 90)` with red fill scaling with progress.
* **Bottom-right**: `GLITCH` overlay — blinking red box when glitch occurs (1 Hz).
* **Center**: translucent ghost icon representing required gesture (for judges only; ensure it is small and off-center to keep "worst UI" feel).
* **All text jitter:** Add small random jitter on position every 600ms (±5 px) so it looks unstable.

# 8 — Alarm scheduling & ringing behavior

* When `CONFIRM` + `THUMBS_UP` is accepted:

  * Validate time (00:00 - 23:59).
  * Convert to next upcoming timestamp (if time is past today, schedule for tomorrow).
  * Save into `alarms.json`.
  * Set internal `Timer` (or apscheduler job).
  * Enter state `ALARM_SET`.

* When alarm triggers:

  * `AlarmProcessor` enters `ALARM_RINGING`.
  * Play loud sound (loop) and overlay flashing visuals on frame.
  * Browser audio plan: implement `AudioPlay` by sending a small JSON message through WebRTC data channel to client JS that plays audio file (agent must implement this interop) — or stream short audio via webrtc track if feasible.
  * To stop: the user must perform `TWO_HANDS_OPEN` (hold 0.8s). On stop, play a success sound and clear the alarm.

# 9 — Persistence & Data formats

* `alarms.json` structure:

```json
[
  {
    "id": "<uuid>",
    "time_str": "11:59",
    "timestamp": 1670000000,
    "created_at": 1670000000,
    "set_by": "gesture_sequence_log_id"
  }
]
```

* Logs: write newline-delimited JSON lines `logs/events.log` with keys: `ts, fps, latency_ms, state, gesture, confidence, event`.

# 10 — Faults, Edge Cases & Fallbacks (must be implemented)

* **False positives**: require hold + confidence threshold.
* **Lighting poor / no detection**: show big overlay: “HAND NOT DETECTED — WINK TO CONTINUE” (blink detection optional) and auto-timeout to previous stable state after `5s`.
* **Multiple hands glitch**: If multiple hands detected, show “TOO MANY HANDS!” overlay and freeze input until single-hand detected.
* **Emergency dev stop**: For developers during demos, allow a hidden keyboard key combination (Ctrl+Shift+S) to stop alarms — not visible to judges; implement via small debug JS accessible only when a debug toggle turned on by a dev environment variable.
* **Browser autoplay restrictions**: implement user gesture or page audio permission during onboarding (agents create a minimal, hidden first click dialog if needed). Keep it hidden but acceptable for demo.

# 11 — Performance & Acceptance Criteria (concrete numbers)

* **Minimum FPS:** 10 FPS in demo environment (30 FPS ideal). Agents should measure and log.
* **Processing latency per frame:** average < 150 ms.
* **Gesture commit success rate (unit tests simulated):** >= 95% on synthetic stable inputs.
* **Glitch behavior:** triggers 10% of the time on commit attempts (exact `GLITCH_P=0.10`).
* **Hold time:** `HOLD_SEC = 2.0s` (configurable).
* **Confidence threshold:** `CONF_THRESH = 0.85`.
* **Swipes detection threshold:** `SWIPE_MIN_DIST_PIX = 70`.

# 12 — Tests and QA checklist (these are the test cases agents must produce)

**Unit tests (simulate landmarks/frames)**:

* `test_fsm_transitions()` — simulate sequence: 1 -> hold -> commit -> 1 -> hold -> commit -> 5 + swipe up -> hold -> commit -> 9 -> hold -> commit -> thumbs_up -> alarm scheduled.
* `test_glitch_rejection()` — ensure 10% of commits randomly reset at 90% progress. (Test by seeding RNG.)
* `test_swipe_direction()` — assert vectors produce correct direction.
* `test_thumb_detection()` — simulate landmarks to produce thumbs-up.
* `test_alarm_ring_stop()` — simulate alarm firing and TWO_HANDS_OPEN stopping.

**Integration tests (headless / CI)**:

* Start container, simulate sample frame stream (agents should create a headless frame generator) and assert the pipeline runs for 60 seconds without crash and logs FPS.

**Manual QA checklist** (for humans):

* Walk through full set->commit->confirm->alarm ring->stop.
* Test in low, medium, and bright lighting.
* Test with left-handed users (mirroring).
* Test multiple people in frame.

# 13 — Demo script (exact steps to record video + narration)

**Duration:** 2 minutes demo + 1-minute defense.

1. Title slide 5s: “Worst UI Alarm — Gesture Only”
2. Live stream: show laptop webcam (streamlit page) — 10s
3. Setting phone: (voiceover) “I set 07:30” — perform gestures:

   * Hold 0: SET_HOUR_TENS (show 0)
   * Hold 7 (5 + swipe right) -> shows 07
   * Hold 3,5 etc. — show commit animations — 30s
4. Show confirmation glitch: attempt commit that resets at 90% (10s).
5. Successfully confirm with `thumbs_up` — schedule alarm — 10s.
6. Fast-forward to alarm ring or wait — show ring visuals — 10s.
7. Stop alarm via `TWO_HANDS_OPEN` — 10s.
8. Close with a line: “We intentionally made the AI punish you — to demonstrate how bad interactions feel.” — 5s.
9. End card with tech stack and judge bullets — 10s.

# 14 — Presentation talking points (exact lines)

* “Technically robust: MediaPipe hands + streamlit-webrtc for real-time video, FSM-driven logic, and precise gesture heuristics.”
* “Design research: exploring how ‘intelligence’ can reduce usability; the project is a satire.”
* “Engineering: OOP processor for deterministic behavior, test suite, and CI.”
* Danger mitigation: “We included dev-only emergency stop, audio permission handling, and logging for reproducibility.”

# 15 — Deployment / Run instructions (what agents must create)

* `Dockerfile` building Python environment with `requirements.txt`.
* Default server port `8501`.
* Entrypoint: `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`.
* `docker run -p 8501:8501 <image>`.
* Add `EXPOSE 8501` in Docker.

# 16 — CI / Automation tasks for agents (discrete tasks for Antigravity)

Give agents these concrete jobs (each job produces artifacts):

1. **Task A — scaffold project**

   * Create repo structure and base files listed in Deliverables.
   * Output: commit to repo branch `scaffold`.

2. **Task B — implement `AlarmProcessor` skeleton**

   * Implement class with `recv` stub and documented stubs for each helper: `count_fingers`, `detect_swipe`, `commit_digit`, `handle_fsm`, `draw_ui`, `schedule_alarm`, `start_ring`, `stop_ring`.
   * No UI CSS here.

3. **Task C — implement gesture detection heuristics**

   * Implement finger count algorithm per spec with EWMA smoothing and persistence frames.
   * Implement swipe detection, thumbs-up, two-hands open.
   * Output: unit tests with synthetic landmarks.

4. **Task D — implement FSM**

   * Implement state machine logic with exact transition table and test harness to simulate sequences.

5. **Task E — UI overlays & CSS**

   * Implement CSS injection and burned-in overlays. Add jitter and glitch visuals.

6. **Task F — alarm scheduler & ring**

   * Implement scheduling using `threading.Timer` or `apscheduler`. Implement ring overlay and client audio play shim.

7. **Task G — tests & CI**

   * Write tests per Test Section and `ci.yml` (run tests, lint, build docker image).

8. **Task H — docs & demo script**

   * Build `README.md`, `presentation_script.md`, and `safety_note.md`.

9. **Task I — packaging**

   * Create Dockerfile and `requirements.txt`.

10. **Task J — final integration & smoke run**

    * Run app in container, run an automated smoke script that simulates key gesture sequences and report metrics.

For each Task, agents must:

* Provide a short commit message.
* Produce unit test coverage report (line coverage).
* Produce a short automation log with fps and latency from smoke-run.

# 17 — Security, Privacy, Accessibility & Ethical Notes (must be included in repo)

* **Privacy**: warn that webcam frames are processed locally; nothing should be uploaded off-device. Agents should ensure no external requests.
* **Permissions**: inform user to accept camera and audio permissions once (browser).
* **Accessibility**: explicitly state that this UI is intentionally inaccessible; include a dev-only accessible alternative (keyboard hotkey) for emergencies.
* **Safety**: no remote code execution of arbitrary user files; sandbox the web app.

# 18 — Acceptance criteria & final sign-off checklist (for agents to complete)

* [ ] All files from Deliverables present in repo.
* [ ] Unit & integration tests passing in CI.
* [ ] Docker image builds successfully.
* [ ] Smoke-run demonstrates: set alarm via gestures, alarm rings, stop via gestures.
* [ ] Telemetry report generated (FPS >= 10, avg latency < 150 ms).
* [ ] Demo video script annotated and ready.
* [ ] README contains a “How to Demo” section.
* [ ] `safety_note.md` included.

# 19 — Extra “Worst UI” finishing touches (small but impactful)

* Randomly change color palette every 7–12 seconds.
* Rotate the large time text by ±2–4 degrees occasionally.
* Randomly swap `LEFT` and `RIGHT` labels (for confusion).
* Pop-up fake success messages: “Alarm Set — (Actually not)”.
* Keep these toggles in `config.json` so judges can flip them off during evaluation if needed.

# 20 — Exact config / constants file (`config.json`) (agents must produce)

```json
{
  "HOLD_SEC": 2.0,
  "HOLD_CONFIRM_SEC": 0.8,
  "CONF_THRESH": 0.85,
  "GLITCH_P": 0.10,
  "SWIPE_MIN_DIST_PIX": 70,
  "PERSIST_FRAMES": 3,
  "EWMA_ALPHA": 0.6,
  "HAND_LOST_TIMEOUT": 5,
  "MODIFIER_WINDOW": 1.2,
  "FPS_TARGET": 15
}
```

# 21 — Final notes for the Antigravity agent chain

* Each agent should *produce testable output*, not just code: unit tests and a short log of a smoke-run.
* Use reproducible seeds for randomized behaviors in tests (`seed=42`).
* Keep camera processing local (no cloud calls).
* Add a developer-mode flag `DEV_MODE=true` that enables emergency keyboard stop and verbose logs.
* Tag PRs with `feature/*` and produce a short PR description that references which deliverable was completed.
