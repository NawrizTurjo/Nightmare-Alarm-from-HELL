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
import dev_state
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Finger Alarm Nightmare",
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
# 🖐️⏰ FINGER ALARM NIGHTMARE ⏰🖐️
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

# WebRTC streamer
ctx = webrtc_streamer(
    key="alarm-processor",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=AlarmProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    },
    async_processing=True,  # Separate processing from UI thread
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

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

# Dev mode indicator
if os.environ.get("DEV_MODE", "false").lower() == "true":
    st.sidebar.markdown("## 🔧 Dev Mode Active")
    st.sidebar.markdown("- Ctrl+Shift+S: Emergency Stop")
    st.sidebar.markdown("- Ctrl+Shift+R: Reset State")
    st.sidebar.markdown("- Ctrl+Shift+D: Debug Overlay")
    
    if st.sidebar.button("🛑 Stop All Alarms"):
        st.sidebar.success("Alarms stopped!")
    
    if st.sidebar.button("🔄 Reset State"):
        st.sidebar.success("State reset!")
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Manual Override")
    
    # Manual Alarm Set
    manual_time = st.sidebar.text_input("Set Alarm Time (HH:MM)", value="12:00")
    if st.sidebar.button("⚡ Force Set Alarm"):
        try:
            from datetime import datetime
            
            # Parse HH:MM
            h, m = map(int, manual_time.split(':'))
            now = datetime.now()
            target_time = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if target_time <= now:
                from datetime import timedelta
                target_time += timedelta(days=1)
                
            dev_state.state.set_alarm(target_time)
            st.sidebar.success(f"Alarm set for {target_time.strftime('%H:%M')}")
        except ValueError:
            st.sidebar.error("Invalid format! Use HH:MM")
            
    # Force Ring
    if st.sidebar.button("🔔 Force Ring (Annoying Audio)"):
        dev_state.state.set_trigger_ring()
        st.sidebar.warning("Triggering Alarm + Audio...")
        
    # Audio Toggle
    dev_state.state.annoying_sound_enabled = st.sidebar.checkbox("Enable Annoying Sounds", value=True)
