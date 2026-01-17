"""
audio_manager.py - WebRTC Audio Source Generator
"""
import os
import random
import logging
import numpy as np
import av
from pathlib import Path
from typing import Optional, List
from pydub import AudioSegment
from pydub.generators import Sine
import threading

logger = logging.getLogger(__name__)

class AudioFrameGenerator:
    """
    Generates AudioFrames for WebRTC streaming.
    Loads MP3s, resamples to 48kHz stereo, and yields frames.
    """
    def __init__(self):
        self.sample_rate = 48000
        self.channels = 2 # Stereo
        self.music_files: List[Path] = []
        self._audio_cache: dict = {} # Cache for pre-loaded audio
        self._scan_music_files()
        
        # Playback state
        self.lock = threading.Lock()
        self.current_audio: Optional[AudioSegment] = None
        self.position_ms = 0
        self.is_playing = False
        self.volume = 0.0 # 0.0 to 1.0 (gain)
        
        # Pre-generated silence frame (20ms)
        # 48000 Hz * 0.02s = 960 samples
        self.frame_duration_ms = 20
        self.samples_per_frame = int(self.sample_rate * (self.frame_duration_ms / 1000))
        
    def _scan_music_files(self):
        audio_dir = Path(__file__).parent / "assets" / "audio"
        if audio_dir.exists():
            exts = ['*.mp3', '*.wav', '*.ogg']
            self.music_files = []
            for ext in exts:
                self.music_files.extend(list(audio_dir.glob(ext)))
            logger.info(f"Found {len(self.music_files)} music files.")
            
            # Pre-load files into memory to prevent lag
            for mp3_path in self.music_files:
                path_str = str(mp3_path)
                if path_str not in self._audio_cache:
                    try:
                        logger.info(f"Pre-loading: {mp3_path.name}")
                        audio = AudioSegment.from_file(path_str)
                        # Pre-convert to 48kHz Stereo for WebRTC
                        audio = audio.set_frame_rate(self.sample_rate).set_channels(self.channels).set_sample_width(2)
                        self._audio_cache[path_str] = audio
                    except Exception as e:
                         logger.error(f"Failed to cache {mp3_path.name}: {e}")

    def _get_fallback_beep(self) -> AudioSegment:
        """Generate a synthesized fallback beep (Sine wave)"""
        # 1000Hz beep for 1 second
        return Sine(1000).to_audio_segment(duration=1000).set_frame_rate(self.sample_rate).set_channels(self.channels).set_sample_width(2)

    def start_alarm(self):
        """Pick a track and start playing"""
        # Rescan (check for new files)
        self._scan_music_files()
        
        with self.lock:
            # Decide what to play
            selected_audio = None
            
            if self.music_files:
                # Try to pick a random file
                try:
                    track_path = random.choice(self.music_files)
                    path_str = str(track_path)
                    
                    # Use cached version if available
                    if path_str in self._audio_cache:
                         selected_audio = self._audio_cache[path_str]
                         logger.info(f"Playing from cache: {track_path.name}")
                    else:
                        logger.info(f"Cache miss, loading: {track_path.name}")
                        audio = AudioSegment.from_file(path_str)
                        selected_audio = audio.set_frame_rate(self.sample_rate).set_channels(self.channels).set_sample_width(2)
                        self._audio_cache[path_str] = selected_audio
                        
                except Exception as e:
                    logger.error(f"Failed to load user audio: {e}")
            
            # Use fallback if nothing loaded
            if selected_audio is None:
                logger.info("Using fallback synthesized beep.")
                selected_audio = self._get_fallback_beep()
            
            self.current_audio = selected_audio
            self.position_ms = 0
            self.volume = 0.1 # Start low
            self.is_playing = True

    def stop_alarm(self):
        with self.lock:
            self.is_playing = False
            self.current_audio = None

    def update_volume(self, dt: float):
        """Ramp volume"""
        if not self.is_playing:
            return
            
        # Ramp up to 1.0 over ~30 seconds
        ramp_rate = 0.03
        with self.lock:
            if self.volume < 1.0:
                self.volume = min(1.0, self.volume + (ramp_rate * dt))

    def get_next_frame(self) -> Optional[av.AudioFrame]:
        """
        Produce the next 20ms audio frame.
        Called by the WebRTC audio processor callback.
        """
        with self.lock:
            if not self.is_playing or not self.current_audio:
                # Return silence
                return self._create_silence()
            
            # Extract chunk
            end_pos = self.position_ms + self.frame_duration_ms
            chunk = self.current_audio[self.position_ms:end_pos]
            
            # Loop if finished
            if len(chunk) < self.frame_duration_ms:
                self.position_ms = 0
                chunk = self.current_audio[0:self.frame_duration_ms]
            else:
                self.position_ms = end_pos
            
            # Convert to numpy for volume adjustment
            # pydub raw data is bytes (int16)
            samples = np.frombuffer(chunk.raw_data, dtype=np.int16)
            
            # Apply volume (simple scalar multiplication)
            # Casting to float for math, then clipping back to int16
            samples = samples.astype(np.float32) * self.volume
            samples = np.clip(samples, -32768, 32767).astype(np.int16)
            
            # Create AudioFrame using PyAV
            # Reshape to (layout, samples) -> Stereo usually (2, N) or interleaved?
            # PyAV expects (samples, channels) for formatting, but from_ndarray expects:
            # layout='stereo' -> shape (samples, 2)
            # pydub gives interleaved samples [L, R, L, R...]
            
            # Reshape for PyAV: (1, total_samples)
            # Interleaved data is already flat-ish, just need 2D wrapper
            flat_samples = samples.reshape(1, -1)
            
            # Create frame
            frame = av.AudioFrame.from_ndarray(flat_samples, format='s16', layout='stereo')
            frame.sample_rate = self.sample_rate
            frame.pts = None # Allow streamer to handle timing
            return frame

    def _create_silence(self) -> av.AudioFrame:
        """Create a silent frame"""
        # Create silent 20ms frame
        # For packed formats (s16), PyAV expects (1, samples * channels)
        # Create (1, 960*2) array
        total_samples = self.samples_per_frame * self.channels
        data = np.zeros((1, total_samples), dtype=np.int16)
        
        frame = av.AudioFrame.from_ndarray(data, format='s16', layout='stereo')
        frame.sample_rate = self.sample_rate
        return frame
