"""
dev_state.py - Shared state for Dev Mode communication
"""
from typing import Optional
from datetime import datetime
import threading

# Shared state class to communicating between Streamlit UI thread and VideoProcessor
class DevState:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending_alarm_time: Optional[datetime] = None
        self.trigger_ring: bool = False
        self.stop_alarm: bool = False # Cheater mode flag
        self.reset_requested: bool = False
        self.annoying_sound_enabled: bool = True
        self.timezone_offset: float = 6.0 # Default UTC+6 (Dhaka)

    def set_alarm(self, dt: datetime):
        with self.lock:
            self.pending_alarm_time = dt
    
    def get_pending_alarm(self) -> Optional[datetime]:
        with self.lock:
            if self.pending_alarm_time:
                dt = self.pending_alarm_time
                self.pending_alarm_time = None
                return dt
            return None

    def set_trigger_ring(self):
        with self.lock:
            self.trigger_ring = True
    
    def check_trigger_ring(self) -> bool:
        with self.lock:
            if self.trigger_ring:
                self.trigger_ring = False
                return True
            return False

# Global instance
state = DevState()
global_audio_generator = None
