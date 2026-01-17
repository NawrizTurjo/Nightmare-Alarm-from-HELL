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
try:
    import winsound
except ImportError:
    winsound = None
from audio_manager import AudioFrameGenerator
import dev_state

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
    hand_count: int = 1
    hand_counts: Optional[List[int]] = None # List of finger counts for all hands found


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
        
        # Use timezone aware "now"
        offset = dev_state.state.timezone_offset
        now = datetime.utcnow() + timedelta(hours=offset)
        
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
        
        # Hold interaction state
        self.five_finger_start_time: Optional[float] = None
        self.roulette_last_change: float = 0.0
        self.current_roulette_digit: Optional[int] = None
        
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
        
        # Alarm stop challenge
        self.stop_challenge: Optional[dict] = None
        
        # Performance tracking
        self.frame_times: List[float] = []
        self.last_frame_time: float = time.time()
        
        # Dev mode
        self.dev_mode = os.environ.get("DEV_MODE", "false").lower() == "true"
        
        # Audio
        # Use AudioFrameGenerator for WebRTC streaming
        from audio_manager import AudioFrameGenerator
        self.audio_manager = AudioFrameGenerator()
        
        # Share with global state so App can access it
        dev_state.global_audio_generator = self.audio_manager
        
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
                f.write(json.dumps(entry, default=str) + '\n')
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
    

    
    # def detect_thumbs_up(self, hand_landmarks, handedness: str) -> Tuple[bool, float]:
    #     """
    def detect_thumbs_up(self, hand_landmarks, handedness: str) -> Tuple[bool, float]:
        """
        Detect thumbs-up gesture using relative height.
        Key Heuristic: Thumb tip should be the highest point (lowest Y) on the hand.
        """
        if hand_landmarks is None:
            return False, 0.0
        
        landmarks = hand_landmarks.landmark
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        index_mcp = landmarks[5]
        
        # 1. Thumb tip must be above (lower Y) the index knuckle (MCP)
        thumb_above_knuckle = thumb_tip.y < index_mcp.y
        
        # 2. Thumb tip should be the highest tip (lowest Y value)
        # We give a small margin of error (0.02)
        is_highest = (
            thumb_tip.y < (index_tip.y - 0.02) and
            thumb_tip.y < (middle_tip.y - 0.02) and
            thumb_tip.y < (ring_tip.y - 0.02) and
            thumb_tip.y < (pinky_tip.y - 0.02)
        )
        
        # 3. Orientation check: Thumb IP (3) should be below Thumb Tip (4)
        thumb_ip = landmarks[3]
        is_upright = thumb_tip.y < thumb_ip.y
        
        is_thumbs_up = thumb_above_knuckle and is_highest and is_upright
        
        return is_thumbs_up, 1.0 if is_thumbs_up else 0.0
    
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
        current_time = time.time()
        
        # Check for two hands open (for alarm stopping)
        if self.detect_two_hands_open(results):
            return GestureResult(
                gesture_type=GestureType.TWO_HANDS_OPEN,
                confidence=0.9,
                is_stable=True,
                hand_count=2
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
        
        # Check for thumbs up gesture (especially important for CONFIRM state)
        is_thumbs_up, thumbs_conf = self.detect_thumbs_up(hand_landmarks, handedness)
        if is_thumbs_up:
            hand_count = 1
            if len(results.multi_hand_landmarks) > 1:
                hand2 = results.multi_hand_landmarks[1]
                handedness2 = "Left" 
                if results.multi_handedness and len(results.multi_handedness) > 1:
                     handedness2 = results.multi_handedness[1].classification[0].label
                
                is_thumbs_up2, _ = self.detect_thumbs_up(hand2, handedness2)
                if is_thumbs_up2:
                    hand_count = 2

            return GestureResult(
                gesture_type=GestureType.THUMBS_UP,
                confidence=thumbs_conf,
                is_stable=True,
                hand_count=hand_count
            )
        
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
        
        # Handle 5-finger gesture - randomize between valid digits 5-9
        if smoothed_count == 5:
            if self.five_finger_start_time is None:
                self.five_finger_start_time = current_time
            
            # Determine valid digits in range 5-9 for current state
            valid_digits = [d for d in [5, 6, 7, 8, 9] if self.is_valid_digit(d)]
            
            if not valid_digits:
                # No valid digits in 5-9 range, just show 5 as feedback
                return GestureResult(
                    gesture_type=GestureType.FINGER_COUNT,
                    finger_count=5,
                    confidence=0.85,
                    is_stable=is_stable
                )
            elif valid_digits == [5]:
                # Only 5 is valid (e.g., SET_MIN_TENS) - no randomization needed
                self.current_roulette_digit = 5
                return GestureResult(
                    gesture_type=GestureType.FINGER_COUNT,
                    finger_count=5,
                    confidence=0.85,
                    is_stable=is_stable
                )
            else:
                # Multiple valid digits - randomize between them
                # Update roulette every 0.15 seconds
                if current_time - self.roulette_last_change > 0.15:
                    self.current_roulette_digit = random.choice(valid_digits)
                    self.roulette_last_change = current_time
                
                # Ensure we have a value (first run)
                if self.current_roulette_digit is None or self.current_roulette_digit not in valid_digits:
                    self.current_roulette_digit = random.choice(valid_digits)
                
                proposed_digit = self.current_roulette_digit
                
                # Map digit to gesture type for visual feedback
                if proposed_digit == 5:
                    proposed_gesture_type = GestureType.FINGER_COUNT
                elif proposed_digit == 6:
                    proposed_gesture_type = GestureType.SWIPE_UP
                elif proposed_digit == 7:
                    proposed_gesture_type = GestureType.SWIPE_RIGHT
                elif proposed_digit == 8:
                    proposed_gesture_type = GestureType.SWIPE_LEFT
                else:
                    proposed_gesture_type = GestureType.SWIPE_DOWN
                
                return GestureResult(
                    gesture_type=proposed_gesture_type,
                    finger_count=proposed_digit if proposed_digit == 5 else 5,
                    confidence=0.85,
                    is_stable=is_stable
                )
            
        # Detect Release (Transition from 5 -> not 5)
        elif self.five_finger_start_time is not None:
            # User just released 5 fingers
            duration = current_time - self.five_finger_start_time
            self.five_finger_start_time = None
            
            # Commit based on duration
            digit = None
            if 2.0 <= duration < 3.0:
                digit = 5
            elif duration >= 3.0:
                # Commit the current random digit shown on screen
                # If user released while seeing "7", they get "7"
                digit = self.current_roulette_digit
                
            # Clear roulette state
            self.current_roulette_digit = None
                
            if digit is not None:
                # CRITICAL Fix: Enforce validation on release commit too!
                if self.is_valid_digit(digit):
                    self.commit_digit(digit)
                # Return NONE this frame to clear state
                return GestureResult(gesture_type=GestureType.NONE)
        
        # Start modifier window for 5 fingers
        if smoothed_count == 5:
            if self.open_palm_start is None:
                self.open_palm_start = time.time()
                self.pending_modifier = True
        else:
            self.open_palm_start = None
            self.pending_modifier = False
        
        # Check second hand for match (for "Double" challenges) and Mixes
        hand_count_match = 1
        conf_boost = 0.0
        all_counts = [int(smoothed_count)] # Primary is smoothed
        
        if len(results.multi_hand_landmarks) > 1:
            hand2 = results.multi_hand_landmarks[1]
            handedness2 = "Left" # default
            if results.multi_handedness and len(results.multi_handedness) > 1:
                 handedness2 = results.multi_handedness[1].classification[0].label
            
            count2 = self.count_fingers(hand2, handedness2)
            all_counts.append(count2)
            
            if count2 == smoothed_count:
                hand_count_match = 2
                conf_boost = 0.1

        # Return finger count or FIST
        final_type = GestureType.FIST if smoothed_count == 0 else GestureType.FINGER_COUNT
        
        return GestureResult(
            gesture_type=final_type,
            finger_count=smoothed_count,
            confidence=(0.85 if is_stable else 0.6) + conf_boost,
            is_stable=is_stable,
            hand_count=hand_count_match,
            hand_counts=sorted(all_counts)
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
        
        # Calculate delay based on same timezone reference
        offset = dev_state.state.timezone_offset
        now = datetime.utcnow() + timedelta(hours=offset)
        delay = (alarm_dt - now).total_seconds()
        
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
        self.is_ringing = True
        self.state = State.ALARM_RINGING
        self._log_event("ALARM_RING")
        
        # Generate annoying stop challenge
        # Generate annoying stop challenge (Single vs Double Hand variations)
        base_challenges = [
            {"type": GestureType.TWO_HANDS_OPEN, "count": 0, "name": "SURRENDER (Open Hands)", "hands": 2},
            {"type": GestureType.THUMBS_UP, "count": 0, "name": "THUMBS UP", "hands": 1},
            {"type": GestureType.THUMBS_UP, "count": 0, "name": "DOUBLE THUMBS UP", "hands": 2},
            {"type": GestureType.FIST, "count": 0, "name": "FIST", "hands": 1},
            {"type": GestureType.FIST, "count": 0, "name": "DOUBLE FIST", "hands": 2},
            {"type": GestureType.FINGER_COUNT, "count": 1, "name": "ONE FINGER", "hands": 1},
            {"type": GestureType.FINGER_COUNT, "count": 1, "name": "DOUBLE ONE FINGER", "hands": 2},
            {"type": GestureType.FINGER_COUNT, "count": 2, "name": "PEACE SIGN", "hands": 1},
            {"type": GestureType.FINGER_COUNT, "count": 2, "name": "DOUBLE PEACE SIGN", "hands": 2},
            {"type": GestureType.FINGER_COUNT, "count": 3, "name": "3 FINGERS", "hands": 1},
            {"type": GestureType.FINGER_COUNT, "count": 3, "name": "DOUBLE 3 FINGERS", "hands": 2},
            {"type": GestureType.FINGER_COUNT, "count": 4, "name": "4 FINGERS", "hands": 1},
            {"type": GestureType.FINGER_COUNT, "count": 4, "name": "DOUBLE 4 FINGERS", "hands": 2},
            {"type": GestureType.FINGER_COUNT, "count": 5, "name": "HIGH FIVE", "hands": 1},
            {"type": GestureType.FINGER_COUNT, "count": 5, "name": "DOUBLE HIGH FIVE", "hands": 2},
            # Mixed Challenges
            {"type": GestureType.FINGER_COUNT, "count": 99, "name": "2 and 3 FINGERS", "hands": 2, "required_counts": [2, 3]},
            {"type": GestureType.FINGER_COUNT, "count": 99, "name": "1 and 5 FINGERS", "hands": 2, "required_counts": [1, 5]},
            {"type": GestureType.FINGER_COUNT, "count": 99, "name": "4 and 0 (Fist)", "hands": 2, "required_counts": [0, 4]},
            {"type": GestureType.FINGER_COUNT, "count": 99, "name": "PEACE and OK (3)", "hands": 2, "required_counts": [2, 3]}, # Duplicate visual but diff text
        ]
        
        chosen = random.choice(base_challenges)
        duration = random.uniform(8.0, 12.0) # Users requested ~10s
        
        text = chosen["name"]
        if chosen["hands"] == 2 and "and" not in text and "DOUBLE" not in text: # Don't double-label
             text += " (BOTH HANDS!)"
        elif chosen["hands"] == 1 and "(One Hand)" not in text:
             text += " (One Hand)"
            
        self.stop_challenge = {
            "type": chosen["type"],
            "count": chosen["count"],
            "text": text,
            "duration": duration,
            "hands": chosen["hands"],
            "required_counts": chosen.get("required_counts")
        }
        self._log_event("CHALLENGE_SET", challenge=self.stop_challenge)
        
        # Start annoying audio in background thread
        if dev_state.state.annoying_sound_enabled:
             # Start MP3 playback (non-blocking)
             self.audio_manager.start_alarm()
             # Also start thread for volume updates/fallback beeps if needed
             threading.Thread(target=self._play_alarm_sound, daemon=True).start()
    
    def _play_alarm_sound(self):
        """Play annoying beep sequence while ringing"""
        freqs = [2000, 4000, 1500, 3000, 500, 5000] # Annoying frequencies
        idx = 0
        while self.is_ringing:
            try:
                # Update volume
                if hasattr(self.audio_manager, 'update_volume'):
                    self.audio_manager.update_volume(0.4) # Approx dt
                
                # Logic: If AudioFrameGenerator is playing music, don't beep.
                if not self.audio_manager.is_playing:
                    # Beep blocks execution, so this thread stays alive
                    if winsound:
                        winsound.Beep(freqs[idx % len(freqs)], 300)
                    else:
                        # Linux/Cloud fallback - just sleep to emulate timing
                        time.sleep(0.3)
                    
                idx += 1
                time.sleep(0.1) # Small gap between beeps
            except Exception as e:
                logger.error(f"Audio error: {e}")
                break

    def stop_alarm(self):
        """Stop the ringing alarm"""
        self.is_ringing = False
        self.state = State.ACK_RING
        self.alarm_time = AlarmTime()
        
        self.audio_manager.stop_alarm()
        
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
        # Check Force Stop (Cheater Mode)
        if self.is_ringing and dev_state.state.stop_alarm:
            self.stop_alarm()
            dev_state.state.stop_alarm = False # Reset flag
            self._log_event("FORCE_STOP_TRIGGERED")
            return
            
        # Check Reset Request
        if dev_state.state.reset_requested:
            self.stop_alarm() # Ensure alarm is off
            # Cancel timer if exists
            if self.alarm_timer:
                self.alarm_timer.cancel()
                self.alarm_timer = None
            
            # Reset all state
            self.state = State.IDLE
            self.alarm_time = AlarmTime()
            self.alarm_scheduled_time = None
            self.current_roulette_digit = None
            self.hold_start_time = None
            self.hold_progress = 0
            
            dev_state.state.reset_requested = False
            self._log_event("SYSTEM_RESET")
            return

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
            # Default challenge (Two hands) if none set (legacy safety)
            target_type = GestureType.TWO_HANDS_OPEN
            target_count = 0
            target_dur = 2.0
            target_hands = 2
            
            if self.stop_challenge:
                target_type = self.stop_challenge["type"]
                target_count = self.stop_challenge["count"]
                target_dur = self.stop_challenge["duration"]
                target_hands = self.stop_challenge.get("hands", 1)
                required_counts = self.stop_challenge.get("required_counts")
            else:
                 required_counts = None
            
            # Check match
            matches = False
            
            # 1. Mixed Gestures Logic
            if required_counts:
                 # Check if detected hand counts match required counts (sorted comparison)
                 if gesture.hand_counts and sorted(gesture.hand_counts) == sorted(required_counts):
                     matches = True
            
            # 2. Standard Logic
            elif gesture.gesture_type == target_type:
                # Check hand count requirements
                if gesture.hand_count >= target_hands:
                    if target_type == GestureType.FINGER_COUNT:
                        if gesture.finger_count == target_count:
                            matches = True
                    else:
                        matches = True
            
            if matches:
                if self.hold_start_time is None:
                    self.hold_start_time = current_time
                
                hold_duration = current_time - self.hold_start_time
                self.hold_progress = hold_duration / target_dur
                
                if hold_duration >= target_dur:
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
                    # For digit 5, we commit on RELEASE only (to allow 6-9 selection)
                    if digit < 5:
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
        
        # Draw hold progress (seconds)
        if self.hold_progress > 0:
            # Assume 2.0s hold for digits (most common case for progress bar visualization)
            # or 0.8s for confirm. Since bar fills up based on progress, we can just show progress * target
            
            # Heuristic: if state is CONFIRM or RINGING, it's 0.8s. Else 2.0s.
            target_sec = CONFIG.get("HOLD_CONFIRM_SEC", 0.8) if self.state in [State.CONFIRM, State.ALARM_RINGING] else CONFIG.get("HOLD_SEC", 2.0)
            elapsed_sec = self.hold_progress * target_sec
            
            pct_text = f"{elapsed_sec:.1f}s"
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
        text = "Show BOTH hands open to stop"
        if self.stop_challenge:
            text = self.stop_challenge["text"]
            
        cv2.putText(
            frame, text,
            (w // 2 - 200, h // 2 + 60),
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
            # Check for Dev Mode commands
            pending_alarm = dev_state.state.get_pending_alarm()
            if pending_alarm:
                 self.alarm_time.hour_tens = pending_alarm.hour // 10
                 self.alarm_time.hour_ones = pending_alarm.hour % 10
                 self.alarm_time.min_tens = pending_alarm.minute // 10
                 self.alarm_time.min_ones = pending_alarm.minute % 10
                 
                 # Go straight to set
                 self.schedule_alarm()
                 self.state = State.ALARM_SET
                 
            if dev_state.state.check_trigger_ring():
                self._trigger_alarm()

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
