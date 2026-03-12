# MeetingMind — Deployment & Audio Device Guide

## 1. Auto-Detection Overview

When the MeetingMind server starts, it runs a one-time device detection scan
(`detect_devices()` in `src/meetingmind/device_detector.py`). The scan:

1. Enumerates every PyAudio-visible audio device on the machine.
2. Categorises each device as **REAL\_MIC**, **VIRTUAL\_MIC**, **SYSTEM\_AUDIO**, or (skipped) output-only.
3. Scores each input device 0–10 based on keyword priority (see table below).
4. Picks the highest-scoring **REAL\_MIC** as the recommended microphone and the highest-scoring **SYSTEM\_AUDIO** device as the recommended loopback.
5. Caches the result and serves it from `GET /devices`.

`POST /meeting/start` uses the auto-detected indices unless the user
explicitly passes `mic_device_index` or `system_device_index` query params.

---

## 2. Device Priority Tiers

| Score | Category | Matched keywords (case-insensitive) |
|------:|----------|--------------------------------------|
| 10 | REAL\_MIC | `array`, `intel`, `realtek`, `built-in`, `internal`, `laptop`, `integrated` |
| 10 | SYSTEM\_AUDIO | `stereo mix`, `loopback`, `what u hear`, `wave out`, `speakers (loopback)` |
| 8 | REAL\_MIC | `usb`, `external` |
| 6 | REAL\_MIC | `headset`, `headphone`, `earphone` |
| 5 | REAL\_MIC | any other device with ≥1 input channel |
| 2 | VIRTUAL\_MIC | `camo`, `obs`, `virtual`, `vb-cable`, `blackhole`, `soundflower`, `voicemeeter`, `screencapture`, `discord`, `zoom mic` |
| — | skipped | `maxInputChannels == 0` (output-only devices) |
| — | skipped | `Microsoft Sound Mapper`, `Primary Sound Capture Driver` (Windows aliases) |

When multiple devices share the top score within a category, the first one
returned by PyAudio is selected.

---

## 3. Platform Notes

### Windows

- **Recommended mic**: Intel® Smart Sound Technology Array (score 10) or any Realtek mic.
- **System audio loopback**: Enable **Stereo Mix** in Windows Sound settings:
  - Right-click the speaker icon → Sound settings → More sound settings
  - Recording tab → right-click → Show Disabled Devices → enable Stereo Mix
- If Stereo Mix is unavailable, install **VB-Audio Cable** (free) and set it as the playback device in Zoom/Teams while using "CABLE Output" as the system source.
- WASAPI host API is preferred; MME devices may appear as duplicates — the detector deduplicates by score.

### macOS

- **Recommended mic**: `Built-in Microphone` (score 10).
- **System audio loopback**: Install **BlackHole** (free, open-source) or **Soundflower**:
  - `brew install blackhole-2ch`
  - Create a Multi-Output Device in Audio MIDI Setup to send audio to both your speakers and BlackHole.
  - BlackHole is categorised as SYSTEM\_AUDIO (score 10); Soundflower as VIRTUAL\_MIC (score 2).

### Linux

- **PulseAudio**: Create a loopback monitor source:
  ```bash
  pactl load-module module-loopback latency_msec=1
  ```
  The resulting `monitor` source will be detected as REAL\_MIC (score 5 or higher depending on name).
- **PipeWire**: Use `pw-loopback` or expose a monitor node.
- If the mic shows score 5 ("unknown"), rename it via PulseAudio so it matches a high-priority keyword.

---

## 4. Manual Override

### Via the web dashboard

The controls bar shows two dropdowns:

- **Mic** — grouped into YOUR MICROPHONE / SYSTEM AUDIO / VIRTUAL. The auto-recommended device is pre-selected with ✅.
- **System** — loopback devices for capturing Zoom/Teams audio. Select "None" if not needed.

Click a different option before pressing **Start Meeting** to override the recommendation.

### Via the REST API

Pass explicit indices as query parameters:

```
POST /meeting/start?model_size=medium&mic_device_index=2&system_device_index=5
```

Device indices are listed by `GET /devices`:

```bash
curl http://localhost:8000/devices | python -m json.tool
```

The response includes `recommended_mic`, `recommended_system`, and `all_devices`
with `index`, `name`, `category`, `score`, `channels`, `sample_rate`, and `host_api`.

---

## 5. Privacy

**All device detection is purely local.**

- Device names are read from the OS via PyAudio.
- No device metadata is stored, logged to a remote service, or transmitted externally.
- Only transcript text is sent externally — to the Anthropic API over HTTPS — and only when you explicitly click **Get PM Suggestions** or call `POST /suggestions`.

See the `PRIVACY` comment at the top of `detect_devices()` in `device_detector.py`.

---

## 6. Troubleshooting

### Mic not listed

- The device has `maxInputChannels == 0` (output-only). Check in your OS sound settings that it has an input channel.
- The device name matches a skip alias (`Microsoft Sound Mapper`, `Primary Sound Capture Driver`). These are OS aliases that duplicate real devices and are intentionally hidden.

### Wrong device auto-selected

- Pass `mic_device_index` or `system_device_index` explicitly.
- Check `GET /devices` → `all_devices` for the correct `index` value.
- If your mic is named generically (e.g. "USB Device"), the detector gives it score 8 for USB or 5 otherwise. A higher-scoring device on the same machine will be preferred.

### Virtual mic capturing silence (Camo, OBS)

- Virtual camera apps such as Camo, OBS, and Discord relay video-only or require the host app to be running. Check that the camera app is active before starting the meeting.
- Select your physical mic from the **Mic** dropdown instead.

### Stereo Mix not available on Windows

- Go to Sound settings → Recording → right-click → Show Disabled Devices.
- If Stereo Mix is still absent, your audio driver may not support it. Install **VB-Audio Virtual Cable** as an alternative.

### Server starts but /devices returns 503

- Device detection runs in a background thread on startup. Wait 1–2 seconds and retry.
- If the error persists, check the server log for a `Device detection failed at startup` message and verify PyAudio is installed (`pip install pyaudio`).

### ffmpeg not on PATH

- Whisper uses ffmpeg to decode audio files passed via the CLI.
- Install ffmpeg and ensure it's on your system PATH, or use `winget install ffmpeg` on Windows.
- Real-time microphone transcription does NOT require ffmpeg.
