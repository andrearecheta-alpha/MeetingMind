# MeetingMind

Real-time meeting transcription.
Privacy-first — runs 100% on your device.

## Features

- **Live transcription** via OpenAI Whisper (runs locally)
- **Speaker identification** — [You] / [Them] labels on every chunk
- **Privacy-first** — audio never leaves your machine
- **Works with any microphone** — auto-detects your default input
- **Mobile accessible** — open the UI from any device on your network
- **RMS level meter** — real-time audio feedback so you know the mic is live
- **Whisper model selector** — choose speed vs. accuracy (tiny → large)

## Tech Stack

Python · FastAPI · Whisper · WebSocket · PyAudio

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/andrearecheta-alpha/MeetingMind.git
cd MeetingMind

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment template
cp .env.example .env

# 4. Start the server
python -m uvicorn meetingmind.main:app --host 0.0.0.0 --port 8000

# 5. Open in your browser
# http://localhost:8000
```

### Windows shortcut

If you installed the `mmstart` CLI entry point:

```powershell
mmstart
```

## Usage

1. Open `http://localhost:8000` in your browser
2. Select your microphone from the dropdown (or leave as OS Default)
3. Choose a Whisper model size (base is recommended for real-time use)
4. Click **Start** — audio is captured immediately, transcript appears once Whisper loads
5. Click **Stop** when the meeting ends
6. Use **Reset** if the server gets stuck in an active state

## Whisper Model Guide

| Model  | Load time | Accuracy | Real-time? |
|--------|-----------|----------|------------|
| tiny   | ~2s       | Low      | Yes        |
| base   | ~5s       | Good     | Yes        |
| small  | ~12s      | Better   | Yes        |
| medium | ~60s      | High     | Marginal   |
| large  | ~100s     | Best     | No         |

The `base` model is the default and works well for clear speech in English.

## Privacy

- Audio is processed entirely on your machine
- Whisper runs locally — no audio is sent to any cloud service
- The WebSocket stream carries transcript text only
- No telemetry, no analytics, no external API calls for transcription

## Requirements

- Python 3.11+
- A working microphone
- ~500 MB disk space for the `base` Whisper model (downloaded once, then cached)
- Windows / macOS / Linux

## Troubleshooting

**"No working microphone found"**
- Windows: Settings → Privacy & Security → Microphone → Allow apps to access your microphone
- Try selecting a specific device from the Mic dropdown instead of OS Default

**Transcript not appearing after model loads**
- Check the RMS meter — if the bar is flat, the mic is silent or muted
- Open `http://localhost:8000/debug` for a full pipeline diagnostic

**Server stuck in "meeting already active"**
- Click **Reset** in the UI, or visit `http://localhost:8000/meeting/reset`

## License

MIT — see [LICENSE](LICENSE)
