"""
test_video_pipeline.py - Integration tests for video processing pipeline

Tests the complete video processing flow including frame handling.
"""

import pytest
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

# Mock mediapipe before importing video_processor
mock_mp = MagicMock()
mock_mp.solutions.hands.Hands.return_value = MagicMock()
mock_mp.solutions.drawing_utils = MagicMock()
sys.modules['mediapipe'] = mock_mp


class TestVideoPipeline:
    """Integration tests for video pipeline"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        from video_processor import AlarmProcessor
        proc = AlarmProcessor()
        # Mock the hands.process method
        proc.hands = MagicMock()
        proc.hands.process.return_value = MagicMock(
            multi_hand_landmarks=None,
            multi_handedness=None
        )
        return proc
    
    def test_pipeline_processes_frame(self, processor):
        """Test that pipeline can process a single frame"""
        # Create a mock frame
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame
        result = processor.recv(mock_frame)
        
        # Should return a numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)
    
    def test_pipeline_handles_no_hands(self, processor):
        """Test pipeline handles frames with no hands detected"""
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process multiple frames
        for _ in range(10):
            result = processor.recv(mock_frame)
        
        # Should not crash
        assert result is not None
    
    def test_fps_calculation(self, processor):
        """Test FPS calculation"""
        # Add some frame times
        processor.frame_times = [0.0, 0.033, 0.066, 0.099, 0.132]
        
        fps = processor._calculate_fps()
        
        # Should be approximately 30 FPS
        assert 25 <= fps <= 40
    
    def test_latency_calculation(self, processor):
        """Test latency calculation"""
        processor.frame_times = [0.0, 0.05]  # 50ms between frames
        
        latency = processor._calculate_latency()
        
        assert latency == pytest.approx(50.0, abs=5)
    
    def test_ui_jitter_updates(self, processor):
        """Test UI jitter offset updates"""
        processor.last_jitter_time = 0
        processor.jitter_offset = (0, 0)
        
        processor.update_ui_effects()
        
        # Jitter should have updated
        assert processor.jitter_offset != (0, 0) or processor.last_jitter_time > 0
    
    def test_color_palette_generation(self, processor):
        """Test random color palette generation"""
        palette = processor._generate_palette()
        
        assert "state_text" in palette
        assert "time_text" in palette
        assert "progress_bar" in palette
        assert "background" in palette
        
        # Colors should be valid BGR tuples
        for color in palette.values():
            assert len(color) == 3
            for c in color:
                assert 0 <= c <= 255


class TestAlarmScheduling:
    """Tests for alarm scheduling functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        from video_processor import AlarmProcessor, AlarmTime
        proc = AlarmProcessor()
        proc.alarm_time = AlarmTime(
            hour_tens=0,
            hour_ones=0,
            min_tens=0,
            min_ones=1
        )
        return proc
    
    def test_alarm_timer_created(self, processor):
        """Test that alarm timer is created on schedule"""
        processor.schedule_alarm()
        
        assert processor.alarm_timer is not None
        assert processor.alarm_scheduled_time is not None
        
        # Clean up
        if processor.alarm_timer:
            processor.alarm_timer.cancel()
    
    def test_alarm_trigger_sets_ringing(self, processor):
        """Test that alarm trigger sets ringing state"""
        processor._trigger_alarm()
        
        assert processor.is_ringing
        from video_processor import State
        assert processor.state == State.ALARM_RINGING
    
    def test_alarm_stop_clears_state(self, processor):
        """Test that stopping alarm clears state"""
        processor.is_ringing = True
        processor.stop_alarm()
        
        assert not processor.is_ringing
        from video_processor import State
        assert processor.state == State.ACK_RING


class TestLogging:
    """Tests for telemetry logging"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        from video_processor import AlarmProcessor
        return AlarmProcessor()
    
    def test_log_event(self, processor, tmp_path):
        """Test that events are logged"""
        import json
        
        # Set log file to temp path
        processor.log_file = tmp_path / "test_events.log"
        
        processor._log_event("TEST_EVENT", extra_data="test")
        
        # Read log file
        with open(processor.log_file, 'r') as f:
            line = f.readline()
        
        entry = json.loads(line)
        assert entry["event"] == "TEST_EVENT"
        assert entry["extra_data"] == "test"


class TestStressRun:
    """Stress tests for stability"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor with mocked MediaPipe"""
        from video_processor import AlarmProcessor
        proc = AlarmProcessor()
        proc.hands = MagicMock()
        proc.hands.process.return_value = MagicMock(
            multi_hand_landmarks=None,
            multi_handedness=None
        )
        return proc
    
    def test_60_second_synthetic_run(self, processor):
        """Run for simulated 60 seconds without crash"""
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Simulate 60 seconds at 15 FPS = 900 frames
        frame_count = 900
        
        for i in range(frame_count):
            result = processor.recv(mock_frame)
            assert result is not None
        
        # Check FPS was tracked
        assert len(processor.frame_times) > 0
        
        fps = processor._calculate_fps()
        assert fps > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
