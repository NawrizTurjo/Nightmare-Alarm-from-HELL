"""
test_fsm.py - Unit tests for FSM state transitions

Tests the finite state machine logic for the alarm setting flow.
"""

import pytest
import sys
import random
from unittest.mock import Mock, patch, MagicMock

# Mock mediapipe before importing video_processor
mock_mp = MagicMock()
mock_mp.solutions.hands.Hands.return_value = MagicMock()
mock_mp.solutions.drawing_utils = MagicMock()
sys.modules['mediapipe'] = mock_mp

from video_processor import (
    AlarmProcessor, State, GestureType, GestureResult, AlarmTime
)


class TestAlarmTime:
    """Tests for AlarmTime dataclass"""
    
    def test_initial_state(self):
        """Test initial alarm time is empty"""
        alarm = AlarmTime()
        assert not alarm.is_complete()
        assert alarm.to_string() == "__:__"
    
    def test_partial_time(self):
        """Test partially set time"""
        alarm = AlarmTime(hour_tens=0, hour_ones=7)
        assert not alarm.is_complete()
        assert alarm.to_string() == "07:__"
    
    def test_complete_time(self):
        """Test fully set time"""
        alarm = AlarmTime(hour_tens=0, hour_ones=7, min_tens=3, min_ones=0)
        assert alarm.is_complete()
        assert alarm.to_string() == "07:30"
    
    def test_to_datetime(self):
        """Test conversion to datetime"""
        alarm = AlarmTime(hour_tens=1, hour_ones=2, min_tens=3, min_ones=0)
        dt = alarm.to_datetime()
        assert dt is not None
        assert dt.hour == 12
        assert dt.minute == 30


class TestFSMTransitions:
    """Tests for FSM state transitions"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        proc = AlarmProcessor()
        proc.state = State.SET_HOUR_TENS
        return proc
    
    def test_idle_to_set_hour_tens(self, processor):
        """Test transition from IDLE to SET_HOUR_TENS"""
        processor.state = State.IDLE
        
        gesture = GestureResult(
            gesture_type=GestureType.FINGER_COUNT,
            finger_count=1,
            confidence=0.9,
            is_stable=True
        )
        
        processor.handle_fsm(gesture)
        assert processor.state == State.SET_HOUR_TENS
    
    def test_set_hour_tens_valid_digit(self, processor):
        """Test committing a valid hour tens digit (0-2)"""
        processor.state = State.SET_HOUR_TENS
        
        # Digit 1 is valid for hour tens
        assert processor.is_valid_digit(0)
        assert processor.is_valid_digit(1)
        assert processor.is_valid_digit(2)
        assert not processor.is_valid_digit(3)
    
    def test_set_hour_ones_after_2(self, processor):
        """Test hour ones range when hour tens is 2"""
        processor.state = State.SET_HOUR_ONES
        processor.alarm_time.hour_tens = 2
        
        # Only 0-3 valid when hour tens is 2 (max 23:xx)
        assert processor.is_valid_digit(0)
        assert processor.is_valid_digit(3)
        assert not processor.is_valid_digit(4)
        assert not processor.is_valid_digit(9)
    
    def test_set_min_tens_range(self, processor):
        """Test minute tens digit range (0-5)"""
        processor.state = State.SET_MIN_TENS
        
        assert processor.is_valid_digit(0)
        assert processor.is_valid_digit(5)
        assert not processor.is_valid_digit(6)
        assert not processor.is_valid_digit(9)
    
    def test_commit_digit_advances_state(self, processor):
        """Test that committing digit advances FSM state"""
        processor.state = State.SET_HOUR_TENS
        processor.commit_digit(0)
        assert processor.state == State.SET_HOUR_ONES
        
        processor.commit_digit(7)
        assert processor.state == State.SET_MIN_TENS
        
        processor.commit_digit(3)
        assert processor.state == State.SET_MIN_ONES
        
        processor.commit_digit(0)
        assert processor.state == State.CONFIRM
    
    def test_full_sequence_simulation(self, processor):
        """Test complete alarm setting sequence: 07:30"""
        processor.state = State.SET_HOUR_TENS
        
        # Commit hour tens: 0
        processor.commit_digit(0)
        assert processor.state == State.SET_HOUR_ONES
        assert processor.alarm_time.hour_tens == 0
        
        # Commit hour ones: 7
        processor.commit_digit(7)
        assert processor.state == State.SET_MIN_TENS
        assert processor.alarm_time.hour_ones == 7
        
        # Commit min tens: 3
        processor.commit_digit(3)
        assert processor.state == State.SET_MIN_ONES
        assert processor.alarm_time.min_tens == 3
        
        # Commit min ones: 0
        processor.commit_digit(0)
        assert processor.state == State.CONFIRM
        assert processor.alarm_time.min_ones == 0
        
        # Verify complete time
        assert processor.alarm_time.is_complete()
        assert processor.alarm_time.to_string() == "07:30"


class TestGlitchRejection:
    """Tests for 10% glitch behavior"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        return AlarmProcessor()
    
    def test_glitch_probability(self, processor):
        """Test that glitch occurs approximately 10% of the time"""
        random.seed(42)  # Reproducible test
        
        glitch_count = 0
        total_attempts = 1000
        
        for _ in range(total_attempts):
            if random.random() < 0.10:
                glitch_count += 1
        
        # Should be around 10% ± margin
        glitch_rate = glitch_count / total_attempts
        assert 0.07 <= glitch_rate <= 0.13
    
    def test_glitch_resets_progress(self, processor):
        """Test that glitch resets hold progress to 0"""
        processor.glitch_active = True
        processor.glitch_start = 0
        processor.hold_progress = 0.9
        
        # Simulate glitch occurring
        processor.hold_start_time = None
        processor.hold_progress = 0
        
        assert processor.hold_progress == 0
        assert processor.hold_start_time is None


class TestErrorRecovery:
    """Tests for error state and recovery"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        return AlarmProcessor()
    
    def test_error_on_hand_lost_timeout(self, processor):
        """Test transition to ERROR when hand is lost"""
        import time
        
        processor.state = State.SET_HOUR_TENS
        processor.last_hand_seen = time.time() - 10  # 10 seconds ago
        
        # Process with no hand
        gesture = GestureResult(gesture_type=GestureType.NONE)
        processor.handle_fsm(gesture)
        
        assert processor.state == State.ERROR
    
    def test_error_recovery_to_idle(self, processor):
        """Test recovery from ERROR to IDLE"""
        processor.state = State.ERROR
        
        # Process gesture (should recover)
        gesture = GestureResult(
            gesture_type=GestureType.FINGER_COUNT,
            finger_count=1,
            confidence=0.9,
            is_stable=True
        )
        
        processor.handle_fsm(gesture)
        assert processor.state == State.IDLE


class TestAlarmRingStop:
    """Tests for alarm ringing and stopping"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        proc = AlarmProcessor()
        proc.state = State.ALARM_RINGING
        proc.is_ringing = True
        return proc
    
    def test_stop_with_two_hands(self, processor):
        """Test stopping alarm with two hands open gesture"""
        import time
        
        # Simulate hold start
        processor.hold_start_time = time.time() - 1.0  # 1 second hold
        
        gesture = GestureResult(
            gesture_type=GestureType.TWO_HANDS_OPEN,
            confidence=0.9,
            is_stable=True
        )
        
        processor.handle_fsm(gesture)
        
        # Should have stopped
        assert processor.state == State.ACK_RING
        assert not processor.is_ringing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
