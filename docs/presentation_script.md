# Presentation Script

## Demo Video (2 minutes)

### Opening (0:00 - 0:10)

**[Title card]**  
"Finger Alarm Nightmare — The Worst Alarm Clock UI"

**[Cut to presenter]**  
"What if your alarm clock... fought back?"

### Live Demo Start (0:10 - 0:25)

**[Show laptop with webcam stream]**  
"This alarm clock has NO buttons. The only way to interact is through hand gestures captured by your webcam."

### Setting the Alarm (0:25 - 1:15)

**[Demonstrate setting 07:30]**

"Let me set an alarm for 7:30 AM."

1. "First, I hold up a fist for the hour tens... zero."
   **[Show fist, wait for loading bar to fill]**
2. "Now for 7... I show five fingers, then swipe right."
   **[Show open palm, swipe right, hold]**
3. "Three for the minute tens..."
   **[Show 3 fingers, hold]**

4. "And zero again for the ones digit."
   **[Show fist, hold]**

### Glitch Moment (1:15 - 1:30)

**[If glitch happens naturally, react]**  
"Oh! It glitched at 90%! There's a 10% chance your input just... fails. Start over!"

**[If no glitch, explain]**  
"And sometimes it randomly fails at 90% progress. Because why should setting an alarm be easy?"

### Confirmation (1:30 - 1:45)

**[Show thumbs up]**  
"To confirm, I give a thumbs up..."
**[Wait for confirmation]**
"Alarm set!"

### Alarm Ring (1:45 - 2:00)

**[Show alarm ringing visuals]**  
"When it rings... the screen goes crazy. And to stop it?"

**[Show two hands open]**  
"You need BOTH hands open. Good luck doing that while half asleep."

### Closing (2:00 - 2:10)

"We built the worst alarm clock to show how AI can punish users — a satire on hostile design."

---

## Judges Defense (1 minute)

### Technical Excellence

"Technically, this is a robust real-time computer vision system:

- MediaPipe Hands for accurate landmark detection
- Finite state machine with 10 states
- streamlit-webrtc for low-latency video processing
- Running at 15+ FPS on a laptop CPU"

### Design Intent

"This is design research disguised as chaos:

- We explored how 'intelligent' interfaces can reduce usability
- The 2-second hold times, 10% failure rate, and gesture-only input create intentional friction
- It's a satire on dark patterns in technology"

### Engineering Rigor

"Despite the chaotic UI:

- Comprehensive test suite with 95%+ coverage
- CI/CD pipeline with Docker deployment
- Structured logging and telemetry
- Emergency failsafes for demos"

### Safety Considerations

"We included dev-mode emergency stops, camera privacy (all local processing), and documented the intentional inaccessibility."

---

## Key Talking Points

1. **"Technically robust, experientially hostile"**
2. **"Real-time CV on consumer hardware"**
3. **"Satire on dark patterns and hostile design"**
4. **"Every frustrating feature was deliberate"**
