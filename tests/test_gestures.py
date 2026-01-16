"""
test_gestures.py - Unit tests for gesture detection

Tests the gesture detection heuristics using synthetic landmarks.
"""

import pytest
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Mock mediapipe before importing video_processor
mock_mp = MagicMock()
mock_mp.solutions.hands.Hands.return_value = MagicMock()
mock_mp.solutions.drawing_utils = MagicMock()
sys.modules['mediapipe'] = mock_mp

from video_processor import AlarmProcessor, GestureType, GestureResult


class MockLandmark:
    """Mock MediaPipe landmark"""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def create_mock_hand_landmarks(finger_states: list, handedness: str = "Right"):
    """
    Create mock hand landmarks for testing.
    
    Args:
        finger_states: List of 5 booleans [thumb, index, middle, ring, pinky]
                      True = extended, False = folded
        handedness: "Left" or "Right"
    
    Returns:
        Mock hand landmarks object
    """
    # Base positions (in normalized coordinates 0-1)
    # Lower y = higher in image
    wrist_y = 0.8
    mcp_y = 0.6
    pip_y = 0.4
    tip_extended_y = 0.2
    tip_folded_y = 0.5
    
    landmarks = []
    
    # Wrist (0)
    landmarks.append(MockLandmark(0.5, wrist_y))
    
    # Thumb landmarks (1-4)
    thumb_mcp_x = 0.3 if handedness == "Right" else 0.7
    thumb_ip_x = 0.25 if handedness == "Right" else 0.75
    thumb_tip_x = 0.2 if finger_states[0] else 0.35  # Extended = further out
    if handedness == "Left":
        thumb_tip_x = 0.8 if finger_states[0] else 0.65
    
    landmarks.append(MockLandmark(thumb_mcp_x, 0.65))  # CMC (1)
    landmarks.append(MockLandmark(thumb_mcp_x, 0.55))  # MCP (2)
    landmarks.append(MockLandmark(thumb_ip_x, 0.45))   # IP (3)
    landmarks.append(MockLandmark(thumb_tip_x, 0.35)) # Tip (4)
    
    # Index finger (5-8)
    tip_y = tip_extended_y if finger_states[1] else tip_folded_y
    landmarks.append(MockLandmark(0.4, mcp_y))    # MCP (5)
    landmarks.append(MockLandmark(0.4, pip_y))    # PIP (6)
    landmarks.append(MockLandmark(0.4, 0.3))      # DIP (7)
    landmarks.append(MockLandmark(0.4, tip_y))    # Tip (8)
    
    # Middle finger (9-12)
    tip_y = tip_extended_y if finger_states[2] else tip_folded_y
    landmarks.append(MockLandmark(0.5, mcp_y))    # MCP (9)
    landmarks.append(MockLandmark(0.5, pip_y))    # PIP (10)
    landmarks.append(MockLandmark(0.5, 0.3))      # DIP (11)
    landmarks.append(MockLandmark(0.5, tip_y))    # Tip (12)
    
    # Ring finger (13-16)
    tip_y = tip_extended_y if finger_states[3] else tip_folded_y
    landmarks.append(MockLandmark(0.6, mcp_y))    # MCP (13)
    landmarks.append(MockLandmark(0.6, pip_y))    # PIP (14)
    landmarks.append(MockLandmark(0.6, 0.3))      # DIP (15)
    landmarks.append(MockLandmark(0.6, tip_y))    # Tip (16)
    
    # Pinky (17-20)
    tip_y = tip_extended_y if finger_states[4] else tip_folded_y
    landmarks.append(MockLandmark(0.7, mcp_y))    # MCP (17)
    landmarks.append(MockLandmark(0.7, pip_y))    # PIP (18)
    landmarks.append(MockLandmark(0.7, 0.3))      # DIP (19)
    landmarks.append(MockLandmark(0.7, tip_y))    # Tip (20)
    
    mock = MagicMock()
    mock.landmark = landmarks
    return mock


class TestFingerCount:
    """Tests for finger counting algorithm"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        return AlarmProcessor()
    
    def test_count_zero_fist(self, processor):
        """Test counting 0 fingers (fist)"""
        landmarks = create_mock_hand_landmarks([False, False, False, False, False])
        count = processor.count_fingers(landmarks, "Right")
        assert count == 0
    
    def test_count_one_finger(self, processor):
        """Test counting 1 finger (index only)"""
        landmarks = create_mock_hand_landmarks([False, True, False, False, False])
        count = processor.count_fingers(landmarks, "Right")
        assert count == 1
    
    def test_count_two_fingers(self, processor):
        """Test counting 2 fingers (index + middle)"""
        landmarks = create_mock_hand_landmarks([False, True, True, False, False])
        count = processor.count_fingers(landmarks, "Right")
        assert count == 2
    
    def test_count_three_fingers(self, processor):
        """Test counting 3 fingers"""
        landmarks = create_mock_hand_landmarks([False, True, True, True, False])
        count = processor.count_fingers(landmarks, "Right")
        assert count == 3
    
    def test_count_four_fingers(self, processor):
        """Test counting 4 fingers (no thumb)"""
        landmarks = create_mock_hand_landmarks([False, True, True, True, True])
        count = processor.count_fingers(landmarks, "Right")
        assert count == 4
    
    def test_count_five_fingers(self, processor):
        """Test counting 5 fingers (open palm)"""
        landmarks = create_mock_hand_landmarks([True, True, True, True, True])
        count = processor.count_fingers(landmarks, "Right")
        assert count == 5
    
    def test_count_left_hand(self, processor):
        """Test counting with left hand (mirrored)"""
        landmarks = create_mock_hand_landmarks([True, True, True, True, True], "Left")
        count = processor.count_fingers(landmarks, "Left")
        assert count == 5


class TestSwipeDetection:
    """Tests for swipe direction detection"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        return AlarmProcessor()
    
    def test_swipe_right(self, processor):
        """Test detecting swipe right"""
        # Simulate movement from left to right
        processor.centroid_history = [
            (100, 300),
            (120, 300),
            (150, 300),
            (190, 300),
            (240, 300),
            (300, 300),
        ]
        
        swipe = processor.detect_swipe()
        assert swipe == GestureType.SWIPE_RIGHT
    
    def test_swipe_left(self, processor):
        """Test detecting swipe left"""
        processor.centroid_history = [
            (300, 300),
            (250, 300),
            (200, 300),
            (150, 300),
            (100, 300),
            (50, 300),
        ]
        
        swipe = processor.detect_swipe()
        assert swipe == GestureType.SWIPE_LEFT
    
    def test_swipe_up(self, processor):
        """Test detecting swipe up (y decreases)"""
        processor.centroid_history = [
            (300, 400),
            (300, 350),
            (300, 300),
            (300, 250),
            (300, 200),
            (300, 150),
        ]
        
        swipe = processor.detect_swipe()
        assert swipe == GestureType.SWIPE_UP
    
    def test_swipe_down(self, processor):
        """Test detecting swipe down (y increases)"""
        processor.centroid_history = [
            (300, 150),
            (300, 200),
            (300, 250),
            (300, 300),
            (300, 350),
            (300, 400),
        ]
        
        swipe = processor.detect_swipe()
        assert swipe == GestureType.SWIPE_DOWN
    
    def test_no_swipe_short_movement(self, processor):
        """Test no swipe detected for small movement"""
        processor.centroid_history = [
            (300, 300),
            (310, 300),
            (320, 300),
            (330, 300),
            (340, 300),
            (350, 300),
        ]
        
        swipe = processor.detect_swipe()
        assert swipe is None  # Movement < 70 pixels


class TestThumbsUpDetection:
    """Tests for thumbs-up gesture detection"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        return AlarmProcessor()
    
    def test_thumbs_up_detected(self, processor):
        """Test detecting thumbs-up gesture"""
        # Thumb extended, others folded
        landmarks = create_mock_hand_landmarks([True, False, False, False, False])
        
        is_thumbs, confidence = processor.detect_thumbs_up(landmarks, "Right")
        
        assert is_thumbs
        assert confidence >= 0.75
    
    def test_not_thumbs_up_open_palm(self, processor):
        """Test that open palm is NOT thumbs up"""
        landmarks = create_mock_hand_landmarks([True, True, True, True, True])
        
        is_thumbs, confidence = processor.detect_thumbs_up(landmarks, "Right")
        
        assert not is_thumbs


class TestFistDetection:
    """Tests for fist detection"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        return AlarmProcessor()
    
    def test_fist_detected(self, processor):
        """Test detecting closed fist"""
        landmarks = create_mock_hand_landmarks([False, False, False, False, False])
        
        is_fist = processor.detect_fist(landmarks)
        
        assert is_fist
    
    def test_not_fist_one_finger(self, processor):
        """Test that one finger up is NOT a fist"""
        landmarks = create_mock_hand_landmarks([False, True, False, False, False])
        
        is_fist = processor.detect_fist(landmarks)
        
        assert not is_fist


class TestEWMASmoothing:
    """Tests for exponential weighted moving average smoothing"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        return AlarmProcessor()
    
    def test_ewma_smoothing(self, processor):
        """Test EWMA smoothing reduces jitter"""
        alpha = 0.6
        
        # Simulate noisy readings
        readings = [3, 3, 2, 3, 3, 4, 3, 3, 2, 3]
        
        smoothed = 0
        for r in readings:
            smoothed = alpha * r + (1 - alpha) * smoothed
        
        # Should be close to 3 (the most common value)
        assert 2.5 <= smoothed <= 3.5
    
    def test_persistence_frames(self, processor):
        """Test that gesture requires persistence frames to be stable"""
        processor.gesture_stable_count = 0
        
        # Simulate 3 identical readings
        for _ in range(3):
            processor.gesture_stable_count += 1
        
        assert processor.gesture_stable_count >= 3


class TestDigitMapping:
    """Tests for gesture to digit mapping"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        return AlarmProcessor()
    
    def test_fist_maps_to_zero(self, processor):
        """Test fist gesture maps to digit 0"""
        gesture = GestureResult(gesture_type=GestureType.FIST, finger_count=0)
        
        digit = processor.get_digit_from_gesture(gesture)
        
        assert digit == 0
    
    def test_finger_count_maps_directly(self, processor):
        """Test finger counts 1-5 map to digits 1-5"""
        for i in range(1, 6):
            gesture = GestureResult(
                gesture_type=GestureType.FINGER_COUNT,
                finger_count=i
            )
            digit = processor.get_digit_from_gesture(gesture)
            assert digit == i
    
    def test_swipe_maps_6_through_9(self, processor):
        """Test swipe gestures map to digits 6-9"""
        mappings = [
            (GestureType.SWIPE_UP, 6),
            (GestureType.SWIPE_RIGHT, 7),
            (GestureType.SWIPE_LEFT, 8),
            (GestureType.SWIPE_DOWN, 9),
        ]
        
        for gesture_type, expected_digit in mappings:
            gesture = GestureResult(gesture_type=gesture_type)
            digit = processor.get_digit_from_gesture(gesture)
            assert digit == expected_digit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
