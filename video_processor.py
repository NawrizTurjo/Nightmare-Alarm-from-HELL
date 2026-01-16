"""
video_processor.py - AlarmProcessor for Gesture-Only Alarm Clock

This module contains the main video processing class that handles:
- Real-time gesture detection via MediaPipe Hands
- Finite State Machine for alarm setting flow
- UI overlay rendering on video frames
- Alarm scheduling and ringing
"""

import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase
import json
import av
import time
import random
import threading
import logging
from datetime import datetime, timedelta
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import os

# MediaPipe imports - handle different versions
import mediapipe as mp

try:
    # Try direct access (older versions)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    # Fallback for newer versions (0.10.x+)
    try:
        from mediapipe.python.solutions import hands as mp_hands
        from mediapipe.python.solutions import drawing_utils as mp_drawing
    except ImportError:
        # Emergency fallback or helpful error
        raise ImportError("Could not import mediapipe.solutions. Please verify installation.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config() -> dict:
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("config.json not found, using defaults")
        return {
            "HOLD_SEC": 2.0,
            "HOLD_CONFIRM_SEC": 0.8,
            "CONF_THRESH": 0.85,
            "GLITCH_P": 0.10,
            "SWIPE_MIN_DIST_PIX": 70,
            "PERSIST_FRAMES": 3,
            "EWMA_ALPHA": 0.6,
            "HAND_LOST_TIMEOUT": 5,
            "MODIFIER_WINDOW": 1.2,
            "FPS_TARGET": 15,
        }


CONFIG = load_config()


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class State(Enum):
    """FSM States for alarm setting flow"""
    IDLE = auto()
    SET_HOUR_TENS = auto()
    SET_HOUR_ONES = auto()
    SET_MIN_TENS = auto()
    SET_MIN_ONES = auto()
    CONFIRM = auto()
    ALARM_SET = auto()
    ALARM_RINGING = auto()
    ACK_RING = auto()
    ERROR = auto()


class GestureType(Enum):
    """Types of gestures that can be detected"""
    NONE = auto()
    FINGER_COUNT = auto()
    THUMBS_UP = auto()
    FIST = auto()
    TWO_HANDS_OPEN = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()


@dataclass
class GestureResult:
    """Result of gesture detection"""
    gesture_type: GestureType
    finger_count: int = 0
    confidence: float = 0.0
    is_stable: bool = False


@dataclass
class AlarmTime:
    """Stores the alarm time being set"""
    hour_tens: Optional[int] = None
    hour_ones: Optional[int] = None
    min_tens: Optional[int] = None
    min_ones: Optional[int] = None
    
    def is_complete(self) -> bool:
        return all([
            self.hour_tens is not None,
            self.hour_ones is not None,
            self.min_tens is not None,
            self.min_ones is not None
        ])
    
    def to_string(self) -> str:
        h_tens = str(self.hour_tens) if self.hour_tens is not None else "_"
        h_ones = str(self.hour_ones) if self.hour_ones is not None else "_"
        m_tens = str(self.min_tens) if self.min_tens is not None else "_"
        m_ones = str(self.min_ones) if self.min_ones is not None else "_"
        return f"{h_tens}{h_ones}:{m_tens}{m_ones}"
    
    def to_datetime(self) -> Optional[datetime]:
        if not self.is_complete():
            return None
        hour = self.hour_tens * 10 + self.hour_ones
        minute = self.min_tens * 10 + self.min_ones
        now = datetime.now()
        alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # If time has passed today, schedule for tomorrow
        if alarm_time <= now:
            alarm_time += timedelta(days=1)
        return alarm_time


# =============================================================================
# ALARM PROCESSOR CLASS
# =============================================================================

class AlarmProcessor(VideoProcessorBase):
    """
    Main video processor for the gesture-only alarm clock.
    Handles gesture detection, FSM logic, and UI rendering.
    """
    
    def __init__(self):
        super().__init__()
        
        # MediaPipe setup - use module-level imports
        self.hands = mp_hands.Hands(
            max_num_hands=CONFIG.get("MEDIAPIPE_MAX_HANDS", 2),
            min_detection_confidence=CONFIG.get("MEDIAPIPE_DETECTION_CONF", 0.7),
            min_tracking_confidence=CONFIG.get("MEDIAPIPE_TRACKING_CONF", 0.6),
            model_complexity=1  # Balanced accuracy/speed (was 0)
        )
        self.mp_draw = mp_drawing
        
        # FSM State
        self.state = State.IDLE
        self.alarm_time = AlarmTime()
        
        # Gesture tracking
        self.current_gesture = GestureResult(GestureType.NONE)
        self.hold_start_time: Optional[float] = None
        self.hold_progress: float = 0.0
        self.last_gesture: Optional[GestureResult] = None
        self.gesture_stable_count: int = 0
        self.ewma_finger_count: float = 0.0
        
        # Swipe detection
        self.centroid_history: List[Tuple[float, float]] = []
        self.open_palm_start: Optional[float] = None
        self.pending_modifier: bool = False
        
        # Hand tracking
        self.last_hand_seen: float = time.time()
        self.hands_detected: int = 0
        
        # Alarm management
        self.alarm_timer: Optional[threading.Timer] = None
        self.is_ringing: bool = False
        self.alarm_scheduled_time: Optional[datetime] = None
        
        # UI state
        self.glitch_active: bool = False
        self.glitch_start: float = 0.0
        self.jitter_offset: Tuple[int, int] = (0, 0)
        self.last_jitter_time: float = 0.0
        self.current_palette: Dict[str, Tuple[int, int, int]] = self._generate_palette()
        self.last_palette_change: float = time.time()
        self.next_palette_change: float = random.uniform(
            CONFIG.get("UI_COLOR_CHANGE_MIN_SEC", 7),
            CONFIG.get("UI_COLOR_CHANGE_MAX_SEC", 12)
        )
        self.fake_success_message: Optional[str] = None
        self.fake_success_start: float = 0.0
        
        # Performance tracking
        self.frame_times: List[float] = []
        self.last_frame_time: float = time.time()
        
        # Dev mode
        self.dev_mode = os.environ.get("DEV_MODE", "false").lower() == "true"
        
        # Initialize logging
        self._init_logging()
    
    def _init_logging(self):
        """Initialize telemetry logging"""
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        self.log_file = log_dir / "events.log"
    
    def _log_event(self, event: str, **kwargs):
        """Log an event to the telemetry file"""
        entry = {
            "ts": datetime.now().isoformat(),
            "fps": self._calculate_fps(),
            "latency_ms": self._calculate_latency(),
            "state": self.state.name,
            "event": event,
            **kwargs
        }
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        return len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0] + 0.001)
    
    def _calculate_latency(self) -> float:
        """Calculate frame processing latency"""
        if len(self.frame_times) < 2:
            return 0.0
        return (self.frame_times[-1] - self.frame_times[-2]) * 1000
    
    def _generate_palette(self) -> Dict[str, Tuple[int, int, int]]:
        """Generate a random (terrible) color palette"""
        return {
            "state_text": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            "time_text": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            "progress_bar": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            "background": (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)),
        }
    
    # =========================================================================
    # GESTURE DETECTION
    # =========================================================================
    
    def count_fingers(self, hand_landmarks, handedness: str) -> int:
        """
        Count extended fingers using MediaPipe landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            handedness: "Left" or "Right"
        
        Returns:
            int: Number of extended fingers (0-5)
        """
        if hand_landmarks is None:
            return 0
        
        landmarks = hand_landmarks.landmark
        
        # Tip and PIP/IP indices for each finger
        # Thumb: tip=4, ip=3
        # Index: tip=8, pip=6
        # Middle: tip=12, pip=10
        # Ring: tip=16, pip=14
        # Pinky: tip=20, pip=18
        
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]  # IP for thumb, PIP for others
        
        extended = []
        
        # Thumb (special case - uses X coordinate)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        
        if handedness == "Right":
            # Right hand: thumb extended if tip.x < ip.x (in mirrored view)
            extended.append(thumb_tip.x < thumb_ip.x)
        else:
            # Left hand: thumb extended if tip.x > ip.x (in mirrored view)
            extended.append(thumb_tip.x > thumb_ip.x)
        
        # Other fingers (use Y coordinate - tip above pip means extended)
        for tip_idx, pip_idx in zip(finger_tips[1:], finger_pips[1:]):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            # In image coordinates, lower Y = higher position
            extended.append(tip.y < pip.y)
        
        return sum(extended)
    
    def detect_swipe(self) -> Optional[GestureType]:
        """
        Detect swipe gestures from centroid history.
        
        Returns:
            GestureType: SWIPE_UP/DOWN/LEFT/RIGHT or None
        """
        if len(self.centroid_history) < 6:
            return None
        
        # Get first and last centroids
        start = self.centroid_history[0]
        end = self.centroid_history[-1]
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < CONFIG.get("SWIPE_MIN_DIST_PIX", 70):
            return None
        
        # Determine direction based on angle
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        if -45 <= angle < 45:
            return GestureType.SWIPE_RIGHT
        elif 45 <= angle < 135:
            return GestureType.SWIPE_DOWN  # Note: Y increases downward
        elif -135 <= angle < -45:
            return GestureType.SWIPE_UP
        else:
            return GestureType.SWIPE_LEFT
    
    def detect_thumbs_up(self, hand_landmarks, handedness: str) -> Tuple[bool, float]:
        """
        Detect thumbs-up gesture.
        
        Returns:
            Tuple[bool, float]: (is_thumbs_up, confidence)
        """
        if hand_landmarks is None:
            return False, 0.0
        
        landmarks = hand_landmarks.landmark
        
        # Check if thumb is up (tip above index MCP)
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        
        thumb_above = thumb_tip.y < index_mcp.y
        
        # Check if other fingers are folded
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        folded_count = 0
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            if landmarks[tip_idx].y > landmarks[pip_idx].y:
                folded_count += 1
        
        is_thumbs_up = thumb_above and folded_count >= 3
        confidence = (folded_count / 4) * (1.0 if thumb_above else 0.5)
        
        return is_thumbs_up, confidence
    
    def detect_fist(self, hand_landmarks) -> bool:
        """Detect closed fist (all fingers folded)"""
        if hand_landmarks is None:
            return False
        
        landmarks = hand_landmarks.landmark
        
        # Check all finger tips below their PIPs
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            if landmarks[tip_idx].y < landmarks[pip_idx].y:
                return False
        
        return True
    
    def detect_two_hands_open(self, results) -> bool:
        """Detect two hands both with open palms"""
        if results.multi_hand_landmarks is None:
            return False
        
        if len(results.multi_hand_landmarks) < 2:
            return False
        
        open_count = 0
        for hand_landmarks in results.multi_hand_landmarks:
            # Count extended fingers
            # Simplified: just check if >= 4 fingers extended
            landmarks = hand_landmarks.landmark
            extended = 0
            finger_tips = [8, 12, 16, 20]
            finger_pips = [6, 10, 14, 18]
            
            for tip_idx, pip_idx in zip(finger_tips, finger_pips):
                if landmarks[tip_idx].y < landmarks[pip_idx].y:
                    extended += 1
            
            if extended >= 4:
                open_count += 1
        
        return open_count >= 2
    
    def process_gestures(self, results) -> GestureResult:
        """
        Main gesture processing pipeline.
        
        Returns:
            GestureResult: Detected gesture with confidence
        """
        # Check for two hands open (for alarm stopping)
        if self.detect_two_hands_open(results):
            return GestureResult(
                gesture_type=GestureType.TWO_HANDS_OPEN,
                confidence=0.9,
                is_stable=True
            )
        
        # Process single hand
        if results.multi_hand_landmarks is None or len(results.multi_hand_landmarks) == 0:
            self.hands_detected = 0
            return GestureResult(gesture_type=GestureType.NONE)
        
        self.hands_detected = len(results.multi_hand_landmarks)
        
        # Get first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = "Right"  # Default
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label
        
        # Update centroid history for swipe detection
        wrist = hand_landmarks.landmark[0]
        self.centroid_history.append((wrist.x * 640, wrist.y * 480))  # Approximate pixel coords
        if len(self.centroid_history) > 10:
            self.centroid_history.pop(0)
        
        # Count fingers
        finger_count = self.count_fingers(hand_landmarks, handedness)
        
        # Apply EWMA smoothing
        alpha = CONFIG.get("EWMA_ALPHA", 0.6)
        self.ewma_finger_count = alpha * finger_count + (1 - alpha) * self.ewma_finger_count
        smoothed_count = round(self.ewma_finger_count)
        
        # Check for stability
        if self.last_gesture and self.last_gesture.finger_count == smoothed_count:
            self.gesture_stable_count += 1
        else:
            self.gesture_stable_count = 0
        
        is_stable = self.gesture_stable_count >= CONFIG.get("PERSIST_FRAMES", 3)
        
        # Check for thumbs up
        is_thumbs_up, thumbs_conf = self.detect_thumbs_up(hand_landmarks, handedness)
        if is_thumbs_up and thumbs_conf >= 0.75:
            return GestureResult(
                gesture_type=GestureType.THUMBS_UP,
                confidence=thumbs_conf,
                is_stable=is_stable
            )
        
        # Check for fist
        if self.detect_fist(hand_landmarks):
            return GestureResult(
                gesture_type=GestureType.FIST,
                finger_count=0,
                confidence=0.9,
                is_stable=is_stable
            )
        
        # Check for swipe (when palm is open)
        if smoothed_count == 5 and self.pending_modifier:
            swipe = self.detect_swipe()
            if swipe:
                self.pending_modifier = False
                self.centroid_history.clear()
                return GestureResult(
                    gesture_type=swipe,
                    finger_count=5,
                    confidence=0.85,
                    is_stable=True
                )
        
        # Start modifier window for 5 fingers
        if smoothed_count == 5:
            if self.open_palm_start is None:
                self.open_palm_start = time.time()
                self.pending_modifier = True
        else:
            self.open_palm_start = None
            self.pending_modifier = False
        
        # Return finger count
        return GestureResult(
            gesture_type=GestureType.FINGER_COUNT,
            finger_count=smoothed_count,
            confidence=0.85 if is_stable else 0.6,
            is_stable=is_stable
        )
    
    # =========================================================================
    # FSM LOGIC
    # =========================================================================
    
    def get_digit_from_gesture(self, gesture: GestureResult) -> Optional[int]:
        """Convert gesture to digit value"""
        if gesture.gesture_type == GestureType.FIST:
            return 0
        elif gesture.gesture_type == GestureType.FINGER_COUNT:
            return gesture.finger_count
        elif gesture.gesture_type == GestureType.SWIPE_UP:
            return 6
        elif gesture.gesture_type == GestureType.SWIPE_RIGHT:
            return 7
        elif gesture.gesture_type == GestureType.SWIPE_LEFT:
            return 8
        elif gesture.gesture_type == GestureType.SWIPE_DOWN:
            return 9
        return None
    
    def is_valid_digit(self, digit: int) -> bool:
        """Check if digit is valid for current state"""
        if self.state == State.SET_HOUR_TENS:
            return 0 <= digit <= 2
        elif self.state == State.SET_HOUR_ONES:
            if self.alarm_time.hour_tens == 2:
                return 0 <= digit <= 3
            return 0 <= digit <= 9
        elif self.state == State.SET_MIN_TENS:
            return 0 <= digit <= 5
        elif self.state == State.SET_MIN_ONES:
            return 0 <= digit <= 9
        return False
    
    def commit_digit(self, digit: int):
        """Commit a digit to the current field"""
        if self.state == State.SET_HOUR_TENS:
            self.alarm_time.hour_tens = digit
            self.state = State.SET_HOUR_ONES
        elif self.state == State.SET_HOUR_ONES:
            self.alarm_time.hour_ones = digit
            self.state = State.SET_MIN_TENS
        elif self.state == State.SET_MIN_TENS:
            self.alarm_time.min_tens = digit
            self.state = State.SET_MIN_ONES
        elif self.state == State.SET_MIN_ONES:
            self.alarm_time.min_ones = digit
            self.state = State.CONFIRM
        
        self._log_event("DIGIT_COMMITTED", digit=digit, new_state=self.state.name)
    
    def schedule_alarm(self):
        """Schedule the alarm to ring at the set time"""
        alarm_dt = self.alarm_time.to_datetime()
        if alarm_dt is None:
            return
        
        self.alarm_scheduled_time = alarm_dt
        delay = (alarm_dt - datetime.now()).total_seconds()
        
        if self.alarm_timer:
            self.alarm_timer.cancel()
        
        self.alarm_timer = threading.Timer(delay, self._trigger_alarm)
        self.alarm_timer.start()
        
        self.state = State.ALARM_SET
        self._save_alarm()
        self._log_event("ALARM_SCHEDULED", time=alarm_dt.isoformat())
    
    def _trigger_alarm(self):
        """Called when alarm timer fires"""
        self.is_ringing = True
        self.state = State.ALARM_RINGING
        self._log_event("ALARM_RING")
    
    def stop_alarm(self):
        """Stop the ringing alarm"""
        self.is_ringing = False
        self.state = State.ACK_RING
        self.alarm_time = AlarmTime()
        
        if self.alarm_timer:
            self.alarm_timer.cancel()
            self.alarm_timer = None
        
        self._log_event("ALARM_STOPPED")
        
        # Return to idle after a moment
        threading.Timer(1.0, self._return_to_idle).start()
    
    def _return_to_idle(self):
        """Return to idle state"""
        self.state = State.IDLE
        self.alarm_time = AlarmTime()
    
    def _save_alarm(self):
        """Persist alarm to alarms.json"""
        import uuid
        alarm_data = {
            "id": str(uuid.uuid4()),
            "time_str": self.alarm_time.to_string(),
            "timestamp": self.alarm_scheduled_time.timestamp() if self.alarm_scheduled_time else 0,
            "created_at": time.time(),
            "set_by": "gesture"
        }
        
        alarms_path = Path(__file__).parent / "alarms.json"
        try:
            with open(alarms_path, 'r') as f:
                alarms = json.load(f)
        except:
            alarms = []
        
        alarms.append(alarm_data)
        
        with open(alarms_path, 'w') as f:
            json.dump(alarms, f, indent=2)
    
    def handle_fsm(self, gesture: GestureResult):
        """Main FSM update logic"""
        current_time = time.time()
        
        # Handle hand lost
        if gesture.gesture_type == GestureType.NONE:
            if current_time - self.last_hand_seen > CONFIG.get("HAND_LOST_TIMEOUT", 5):
                if self.state not in [State.IDLE, State.ALARM_SET, State.ALARM_RINGING]:
                    self.state = State.ERROR
                    self._log_event("HAND_LOST")
            return
        
        self.last_hand_seen = current_time
        
        # Handle ringing state
        if self.state == State.ALARM_RINGING:
            if gesture.gesture_type == GestureType.TWO_HANDS_OPEN:
                if self.hold_start_time is None:
                    self.hold_start_time = current_time
                
                hold_duration = current_time - self.hold_start_time
                self.hold_progress = hold_duration / CONFIG.get("HOLD_CONFIRM_SEC", 0.8)
                
                if hold_duration >= CONFIG.get("HOLD_CONFIRM_SEC", 0.8):
                    self.stop_alarm()
                    self.hold_start_time = None
                    self.hold_progress = 0
            else:
                self.hold_start_time = None
                self.hold_progress = 0
            return
        
        # Handle idle -> start setting
        if self.state == State.IDLE:
            if gesture.gesture_type != GestureType.NONE:
                self.state = State.SET_HOUR_TENS
                self._log_event("STATE_CHANGE", new_state=self.state.name)
            return
        
        # Handle confirm state
        if self.state == State.CONFIRM:
            if gesture.gesture_type == GestureType.THUMBS_UP:
                if self.hold_start_time is None:
                    self.hold_start_time = current_time
                
                hold_duration = current_time - self.hold_start_time
                self.hold_progress = hold_duration / CONFIG.get("HOLD_CONFIRM_SEC", 0.8)
                
                if hold_duration >= CONFIG.get("HOLD_CONFIRM_SEC", 0.8):
                    self.schedule_alarm()
                    self.hold_start_time = None
                    self.hold_progress = 0
            else:
                self.hold_start_time = None
                self.hold_progress = 0
            return
        
        # Handle digit setting states
        if self.state in [State.SET_HOUR_TENS, State.SET_HOUR_ONES, 
                          State.SET_MIN_TENS, State.SET_MIN_ONES]:
            digit = self.get_digit_from_gesture(gesture)
            
            if digit is not None and self.is_valid_digit(digit) and gesture.is_stable:
                # Track hold for this digit
                if self.hold_start_time is None:
                    self.hold_start_time = current_time
                    self._log_event("HOLD_START", digit=digit)
                
                hold_duration = current_time - self.hold_start_time
                self.hold_progress = hold_duration / CONFIG.get("HOLD_SEC", 2.0)
                
                # Check for glitch at 90%
                if self.hold_progress >= 0.9 and not self.glitch_active:
                    if random.random() < CONFIG.get("GLITCH_P", 0.1):
                        self.glitch_active = True
                        self.glitch_start = current_time
                        self.hold_start_time = None
                        self.hold_progress = 0
                        self._log_event("GLITCH_TRIGGERED")
                        return
                
                # Complete hold
                if hold_duration >= CONFIG.get("HOLD_SEC", 2.0):
                    self.commit_digit(digit)
                    self.hold_start_time = None
                    self.hold_progress = 0
            else:
                self.hold_start_time = None
                self.hold_progress = 0
        
        # Handle error recovery
        if self.state == State.ERROR:
            # Return to idle after a moment
            self.state = State.IDLE
            self.alarm_time = AlarmTime()
        
        # Clear glitch after 1 second
        if self.glitch_active and current_time - self.glitch_start > 1.0:
            self.glitch_active = False
    
    # =========================================================================
    # UI DRAWING
    # =========================================================================
    
    def update_ui_effects(self):
        """Update UI jitter and color effects"""
        current_time = time.time()
        
        # Update jitter
        if current_time - self.last_jitter_time > 0.6:
            self.last_jitter_time = current_time
            jitter_range = CONFIG.get("UI_JITTER_RANGE_PX", 5)
            self.jitter_offset = (
                random.randint(-jitter_range, jitter_range),
                random.randint(-jitter_range, jitter_range)
            )
        
        # Update color palette
        if current_time - self.last_palette_change > self.next_palette_change:
            if CONFIG.get("WORST_UI_FEATURES", {}).get("color_chaos", True):
                self.current_palette = self._generate_palette()
            self.last_palette_change = current_time
            self.next_palette_change = random.uniform(
                CONFIG.get("UI_COLOR_CHANGE_MIN_SEC", 7),
                CONFIG.get("UI_COLOR_CHANGE_MAX_SEC", 12)
            )
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw all UI overlays on the frame"""
        h, w = frame.shape[:2]
        
        self.update_ui_effects()
        jx, jy = self.jitter_offset
        
        # Draw state label (top-left)
        state_text = f"STATE: {self.state.name}"
        cv2.putText(
            frame, state_text,
            (20 + jx, 40 + jy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            self.current_palette.get("state_text", (0, 0, 255)),
            2
        )
        
        # Draw current time (below state)
        time_text = self.alarm_time.to_string()
        cv2.putText(
            frame, time_text,
            (20 + jx, 100 + jy),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            self.current_palette.get("time_text", (255, 0, 0)),
            3
        )
        
        # Draw progress bar (bottom-left)
        bar_x1, bar_y1 = 50, h - 120
        bar_x2, bar_y2 = 350, h - 90
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
        
        if self.hold_progress > 0:
            fill_width = int((bar_x2 - bar_x1) * min(self.hold_progress, 1.0))
            cv2.rectangle(
                frame,
                (bar_x1, bar_y1),
                (bar_x1 + fill_width, bar_y2),
                self.current_palette.get("progress_bar", (0, 0, 255)),
                -1
            )
        
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 2)
        
        # Draw hold percentage
        pct_text = f"{int(self.hold_progress * 100)}%"
        cv2.putText(
            frame, pct_text,
            (bar_x2 + 10, bar_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Draw glitch overlay (bottom-right)
        if self.glitch_active:
            if int(time.time() * 2) % 2 == 0:  # Blink at 1Hz
                cv2.rectangle(
                    frame,
                    (w - 150, h - 80),
                    (w - 10, h - 30),
                    (0, 0, 255),
                    -1
                )
                cv2.putText(
                    frame, "GLITCH!",
                    (w - 140, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )
        
        # Draw FPS (bottom-right corner)
        fps = self._calculate_fps()
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (w - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1
        )
        
        # Draw alarm ringing overlay
        if self.is_ringing:
            frame = self._draw_ringing_overlay(frame)
        
        # Draw too many hands warning
        if self.hands_detected > 1 and self.state not in [State.ALARM_RINGING, State.ACK_RING]:
            cv2.putText(
                frame, "TOO MANY HANDS!",
                (w // 2 - 150, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )
        
        # Draw hand not detected warning
        if self.state == State.ERROR:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            
            cv2.putText(
                frame, "HAND NOT DETECTED",
                (w // 2 - 200, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3
            )
            cv2.putText(
                frame, "Show your hand to continue",
                (w // 2 - 180, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        
        # Draw alarm set confirmation
        if self.state == State.ALARM_SET and self.alarm_scheduled_time:
            time_str = self.alarm_scheduled_time.strftime("%H:%M")
            cv2.putText(
                frame, f"ALARM SET: {time_str}",
                (w // 2 - 150 + jx, h // 2 + jy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
        
        # Draw gesture hint
        self._draw_gesture_hint(frame)
        
        return frame
    
    def _draw_ringing_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw alarm ringing visual effects"""
        h, w = frame.shape[:2]
        
        # Flashing background
        if int(time.time() * 4) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Shake effect
        shake_x = random.randint(-10, 10)
        shake_y = random.randint(-10, 10)
        
        # WAKE UP text
        cv2.putText(
            frame, "WAKE UP!!!",
            (w // 2 - 150 + shake_x, h // 2 + shake_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            4
        )
        
        # Instructions
        cv2.putText(
            frame, "Show BOTH hands open to stop",
            (w // 2 - 180, h // 2 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        # Stop progress
        if self.hold_progress > 0:
            cv2.putText(
                frame, f"Stopping: {int(self.hold_progress * 100)}%",
                (w // 2 - 100, h // 2 + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        
        return frame
    
    def _draw_gesture_hint(self, frame: np.ndarray):
        """Draw a small ghost gesture hint"""
        h, w = frame.shape[:2]
        
        hints = {
            State.SET_HOUR_TENS: "0-2 fingers",
            State.SET_HOUR_ONES: "0-9 fingers",
            State.SET_MIN_TENS: "0-5 fingers",
            State.SET_MIN_ONES: "0-9 fingers",
            State.CONFIRM: "Thumbs up!",
            State.ALARM_RINGING: "Two hands!",
        }
        
        hint = hints.get(self.state)
        if hint:
            # Draw small, off-center hint (worst UI style)
            cv2.putText(
                frame, hint,
                (w // 2 - 50, h // 2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (128, 128, 128),  # Gray, hard to see
                1
            )
    
    # =========================================================================
    # MAIN FRAME PROCESSING
    # =========================================================================
    
    def recv(self, frame):
        """
        Process each video frame.
        """
        try:
            # Track frame timing
            current_time = time.time()
            self.frame_times.append(current_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            
            # Convert frame using av
            img = frame.to_ndarray(format="bgr24")
            
            # Flip horizontally for mirror effect
            img = cv2.flip(img, 1)
            
            # Process with MediaPipe
            # Note: MediaPipe expects RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Lock to ensure thread safety if needed (though we create hands per instance)
            results = self.hands.process(img_rgb)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
            
            # Process gestures
            gesture = self.process_gestures(results)
            self.last_gesture = gesture
            
            # Update FSM
            self.handle_fsm(gesture)
            
            # Draw UI
            img = self.draw_ui(img)
            
            # Log frame (sample rate)
            if random.random() < 0.05:  # Log 5% of frames
                self._log_event(
                    "FRAME_PROCESSED",
                    gesture=gesture.gesture_type.name,
                    finger_count=gesture.finger_count,
                    confidence=gesture.confidence
                )
            
            # Return processed frame
            from av import VideoFrame
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            # Return original frame on error to prevent freeze
            return frame
