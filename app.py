"""
app.py - Streamlit Entrypoint for Finger Alarm Nightmare

The worst alarm clock UI - controlled entirely by hand gestures.
For the Worst UI Competition.
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import os

# Import the video processor
from video_processor import AlarmProcessor
from audio_manager import AudioFrameGenerator
from streamlit_webrtc import AudioProcessorBase
import dev_state
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Nightmare Alarm from HELL",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CURSED CSS INJECTION
# =============================================================================

CURSED_CSS = """
<style>
    /* Testing Mode - CSS Cleaned */
    body {
        font-family: sans-serif;
    }
    
    .stVideo {
        border: 2px solid #333;
    }
    
    h1 { color: #333; }
</style>
"""

st.markdown(CURSED_CSS, unsafe_allow_html=True)

# =============================================================================
# AUDIO PERMISSION JS SHIM
# =============================================================================

AUDIO_JS = """
<script>
    // Request audio permission on page load
    document.addEventListener('DOMContentLoaded', function() {
        // Create a silent audio context to get permission
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        const audioCtx = new AudioContext();
        
        // Resume on first click
        document.addEventListener('click', function resumeAudio() {
            audioCtx.resume().then(() => {
                console.log('Audio context resumed');
            });
            document.removeEventListener('click', resumeAudio);
        }, { once: true });
    });
    
    // Dev mode emergency stop (Ctrl+Shift+S)
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.shiftKey && e.key === 'S') {
            console.log('Emergency stop triggered');
            // Send message to stop alarm (would need SSE/websocket in production)
            window.postMessage({ type: 'EMERGENCY_STOP' }, '*');
        }
    });
</script>
"""

st.markdown(AUDIO_JS, unsafe_allow_html=True)

# =============================================================================
# MAIN UI
# =============================================================================

# Title with maximum chaos
st.markdown("""
# 🖐️⏰ Nightmare Alarm from HELL ⏰🖐️
### *The alarm clock that HATES you*
""")

# Warning banner
st.markdown("""
<div class="warning-text">
⚠️ NO BUTTONS ALLOWED! Use ONLY hand gestures! ⚠️
</div>
""", unsafe_allow_html=True)

# Instructions (poorly formatted on purpose)
with st.expander("📖 How to Suffer (Instructions)", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Setting Digits
        | Fingers | Digit |
        |---------|-------|
        | ✊ Fist | 0 |
        | ☝️ One | 1 |
        | ✌️ Two | 2 |
        | 🤟 Three | 3 |
        | 🖐️ Four | 4 |
        | 🖐️ Five | 5 |
        """)
    
    with col2:
        st.markdown("""
        ### Higher Digits (6-9)
        | Gesture | Digit |
        |---------|-------|
        | 🖐️ Hold 5 Fingers | 🎰 ROULETTE |
        | (> 3 seconds) | Stops on random # |
        | *Good luck timing it!* | *Ha ha ha* |
        """)
    
    st.markdown("""
    ---
    ### Controls
    - **👍 Thumbs Up** → Confirm alarm (hold 0.8s)
    - **🙌 Stop Alarm** → FOLLOW SCREEN INSTRUCTIONS!
    - **(Challenges vary: One hand, two hands, specific fingers...)**
    - **10% chance of GLITCH** at 90% progress (start over!)
    """)

st.markdown("---")

# Main video stream
st.markdown("### 📹 Gesture Detection Zone")

# Audio Processor Class
class AlarmAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.generator = dev_state.global_audio_generator

    def recv(self, frame):
        # We ignore input audio (mic) and return generated audio
        # or silence if not ringing
        new_frame = self.generator.get_next_frame()
        if new_frame:
            return new_frame
        return frame # Fallback

# WebRTC streamer
try:
    # WebRTC streamer
    ctx = webrtc_streamer(
        key="alarm-processor",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=AlarmProcessor,
        audio_processor_factory=AlarmAudioProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30}
            },
            "audio": True
        },
        async_processing=True,  # Separate processing from UI thread
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]
        }
    )
except Exception as e:
    st.error("WebRTC failed to initialize. Try refreshing or running locally.")
    st.exception(e)
    ctx = None

# Status area
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📍 Status
    *Watch the video overlay for current state*
    """)

with col2:
    st.markdown("""
    ### ⏱️ Progress
    *Loading bar shows gesture hold progress*
    """)

with col3:
    st.markdown("""
    ### 💥 Glitches
    *Random failures are a FEATURE*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.5; font-size: 12px;">
    Built with 💢 for the Worst UI Competition | 
    <a href="docs/safety_note.md">Safety Note</a> |
    Press Ctrl+Shift+S for emergency stop (dev mode only)
</div>
""", unsafe_allow_html=True)

# Dev mode / Cheater Access
if True: # Always active but hidden
    st.sidebar.markdown("---")
    
    # Password Protection
    password = st.sidebar.text_input("Nice Try...ekhn password type koren 😈", type="password", key="cheater_pass")
    
    # Get password from env
    ADMIN_PASSWORD = os.getenv("PASSWORD")
    
    if password:
        if password.strip().casefold() == ADMIN_PASSWORD.strip().casefold():
            st.sidebar.success("Wow! You are a brilliant! 🫨")
            st.sidebar.subheader("🕵️ Cheater Menu")
            
            # --- ADMIN CONTROLS ---
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("🛑 STOP"):
                    dev_state.state.stop_alarm = True
                    st.sidebar.success("STOPPED")
            with col2:        
                if st.button("🔄 RESET"):
                    dev_state.state.reset_requested = True
                    st.sidebar.info("RESET REQUESTED")
            
            st.sidebar.markdown("---")
            
            # Manual Alarm Set
            manual_time = st.sidebar.text_input("Set Alarm (HH:MM)", value="12:00")
            if st.sidebar.button("⚡ Force Set"):
                try:
                    from datetime import datetime
                    h, m = map(int, manual_time.split(':'))
                    now = datetime.now()
                    target_time = now.replace(hour=h, minute=m, second=0, microsecond=0)
                    if target_time <= now:
                        from datetime import timedelta
                        target_time += timedelta(days=1)
                    dev_state.state.set_alarm(target_time)
                    st.sidebar.success(f"Set: {target_time.strftime('%H:%M')}")
                except ValueError:
                    st.sidebar.error("Invalid format!")
                    
            # Force Ring
            if st.sidebar.button("🔔 Force Ring"):
                dev_state.state.set_trigger_ring()
                st.sidebar.warning("Triggering...")

            # Audio Toggle
            dev_state.state.annoying_sound_enabled = st.sidebar.checkbox("Enable Sound", value=True)
            
            # Timezone Config
            st.sidebar.caption("🕒 Timezone: UTC+6")
        else:
            st.sidebar.error("Hehe, nice try! 😒")
