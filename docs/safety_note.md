# Safety & Accessibility Note

## ⚠️ Intentional Design Disclaimer

This application is created for the **"Worst UI Competition"** and is **intentionally designed to be frustrating and inaccessible**. It is a satirical demonstration of how technology can create poor user experiences.

**This is NOT intended for real-world use as an alarm clock.**

## Privacy

- ✅ All webcam frames are processed **locally** on your device
- ✅ No video or image data is uploaded to any server
- ✅ No external API calls are made
- ✅ All data stays on your machine

## Accessibility

This UI is **intentionally inaccessible** and demonstrates anti-patterns:

- ❌ No keyboard navigation
- ❌ No screen reader support
- ❌ Rapidly changing colors (seizure risk)
- ❌ Small, jittering text
- ❌ No alternative input methods
- ❌ High dexterity requirements

### Emergency Developer Controls

For demo safety, the following hidden controls are available when `DEV_MODE=true`:

| Shortcut       | Action                    |
| -------------- | ------------------------- |
| `Ctrl+Shift+S` | Emergency stop all alarms |
| `Ctrl+Shift+R` | Reset to IDLE state       |
| `Ctrl+Shift+D` | Toggle debug overlay      |

**These controls are intentionally hidden from judges.**

## Browser Permissions

The app requires:

- 📷 Camera access (for gesture detection)
- 🔊 Audio playback permission (for alarm sound)

## Health Warning

**Photosensitivity Warning**: This application contains:

- Rapidly flashing colors
- Screen shake effects
- High contrast color changes

If you are sensitive to flashing lights, please use caution.

## Sandbox Environment

- The app runs in browser sandbox
- No system-level file access
- No arbitrary code execution
- Safe to run on personal devices
