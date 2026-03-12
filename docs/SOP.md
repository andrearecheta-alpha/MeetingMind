# MeetingMind — Standard Operating Procedures

Version: 1.0 | Updated: 2026-02-27

---

## SOP-001 — Start the Development Server

**When:** Beginning any development or testing session.

```powershell
# From project root in PowerShell
$env:PYTHONPATH="src"
python -m uvicorn meetingmind.main:app --host 0.0.0.0 --port 8000
```

**Verify:**
- Terminal shows `Application startup complete.`
- Open `http://localhost:8000` → dashboard loads
- Status pill shows "Connecting…" then "Connected"

**If port 8000 is busy:**
```powershell
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

---

## SOP-002 — Run the Full Test Suite

**When:** Before committing any code change. After any sprint is declared complete.

```powershell
$env:PYTHONPATH="src"; python -m pytest tests/ -v
```

**Expected:** All tests pass. Count must match or exceed sprint baseline.
**On failure:** Do not commit. Diagnose and fix before proceeding.

```powershell
# Run a single test file
$env:PYTHONPATH="src"; python -m pytest tests/test_coaching.py -v

# Run with coverage
$env:PYTHONPATH="src"; python -m pytest tests/ --cov=meetingmind --cov-report=term-missing
```

---

## SOP-003 — Debug a Stuck Pipeline

**When:** Transcript not appearing after meeting starts.

**Step 1 — Check /debug endpoint:**
```
http://localhost:8000/debug
```

**Diagnosis table:**

| `chunks_from_queue` | `chunks_transcribed` | Diagnosis |
|---|---|---|
| 0 after 5s | — | PyAudio not capturing — check mic permissions |
| > 0 | 0 | All chunks silenced — mic too quiet or gate too high |
| > 0 | > 0 | Whisper working but WS not connected — check browser |
| > 0 | > 0 | Pipeline OK — check if /ws/transcribe connected |

**Step 2 — Check RMS meter:**
- Flat bar → no audio signal → check mic privacy settings
- Bar moves but no transcript → Whisper filtering → check threshold

**Step 3 — Inject test chunk:**
```
POST http://localhost:8000/debug/inject
```
If this appears in UI → WS pipeline works, issue is upstream (audio or Whisper).

**Step 4 — Test Whisper directly:**
```
POST http://localhost:8000/debug/test-whisper
```

---

## SOP-004 — Reset a Stuck Meeting State

**When:** Server returns "A meeting is already active" on Start click.

**Option A — UI button:**
Click `↺ Reset` in the controls bar.

**Option B — Browser URL bar:**
```
http://localhost:8000/meeting/reset
```

**Option C — curl.exe:**
```powershell
curl.exe -X POST http://localhost:8000/meeting/reset
```

**Verify:** `/health` returns `"meeting_active": false`.

---

## SOP-005 — Push Code to Private Repo

**When:** After any significant code change or sprint completion.

```powershell
# Stage specific files (never use git add -A blindly)
git add src/meetingmind/main.py src/meetingmind/suggestion_engine.py

# Commit
git commit -m "descriptive message"

# Push to private only
git push private main
```

**NEVER push to origin/main from the main branch.**
Public repo (origin) is the stripped version on the `public-strip` branch.

---

## SOP-006 — Verify Private Repo Integrity

**When:** After any push to private, or before a new sprint.

```powershell
git ls-remote private              # confirm HEAD exists
git ls-tree -r --name-only HEAD    # list all tracked files
git ls-tree -r --name-only HEAD | wc -l  # count
```

**Minimum file count:** 47 (as of Sprint 4 end).
**Mandatory files:** `suggestion_engine.py`, `context_engine.py`, `knowledge_base.py`,
`device_detector.py`, `token_budget.py`, `guest_session.py`, `docs/COACHING_PROMPTS.md`,
`docs/PRODUCT_VISION.md`, `docs/ANALYTICS_FRAMEWORK.md`.

---

## SOP-007 — Session Startup Checklist

Run at the beginning of every Claude Code session:

```
□ git branch                          → confirm on main (not public-strip)
□ git log --oneline -3                → note last commit
□ python -m pytest tests/ -v          → confirm baseline passing
□ Review MEMORY.md                    → load current sprint status
□ Review ACTION_ITEMS.md              → know what's planned
□ Review KNOWN_DEFECTS.md             → avoid re-breaking known issues
```

---

## SOP-008 — Session End Checklist

Run before closing every Claude Code session:

```
□ Run full test suite — record count in commit message
□ git add (specific files only)
□ git commit -m "descriptive message with test count"
□ git push private main
□ Update MEMORY.md with current sprint status
□ Update CHECKPOINT_SPRINTn.md if sprint milestone reached
□ Update ACTION_ITEMS.md if new items identified
□ Update KNOWN_DEFECTS.md if new defects found
□ Update RISK_LOG.md if new risks identified
```

---

## SOP-009 — Add a New Endpoint

**Checklist for every new FastAPI endpoint:**

```
□ Define endpoint in main.py with docstring
□ Add to module docstring endpoint table at top of main.py
□ Add test in appropriate test file
□ If endpoint exposes new data, check for sensitive info (API keys, PII)
□ If endpoint calls external API (Claude), add to _api_key.py usage
□ Update MEMORY.md architecture section if endpoint is significant
```

---

## SOP-010 — Add a New Coaching Trigger

1. Add trigger ID and pattern to `COACHING_PATTERNS` in `suggestion_engine.py`
2. Add test case in `tests/test_coaching.py` — minimum: trigger fires, cooldown works
3. Add trigger to `COACHING_PROMPTS.md` with example text and expected outcome
4. Test manually: start meeting, speak trigger phrase, verify coaching panel updates

---

## SOP-011 — Change the Silence Gate Threshold

**When:** Transcription not appearing; chunks_filtered == chunks_from_queue.

**In `transcriber.py`:**
```python
_SILENCE_PEAK_THRESHOLD = 0.001   # lower this value
```

**In `main.py`:**
```python
_SILENCE_THRESHOLD = 0.00005      # RMS marker line in UI meter
```

**Diagnostic sequence:**
1. Start meeting, check `/debug` for `last_rms` value
2. Set `_SILENCE_PEAK_THRESHOLD` to 50% of the observed `last_rms`
3. Restart server, verify chunks pass gate

**Known baseline (Intel laptop mic):** RMS ≈ 0.000075 → threshold set to 0.00005.

---

## SOP-012 — Update the Whisper Model Default

**In `main.py`:**
```python
_WHISPER_MODEL_DEFAULT = "base"    # change here
```

This affects: preload at startup, default for `POST /meeting/start`.
After changing, restart the server. The model downloads on first use (~5 min for medium).

---

## SOP-013 — Diagnose WebSocket Disconnections

**Symptoms:** Status shows "Disconnected"; transcript stops appearing mid-meeting.

**Check 1 — Browser console:**
Open DevTools → Console → look for `WebSocket connection closed` errors.

**Check 2 — Server logs:**
```
WebSocket client disconnected. Active connections: 0
```

**Check 3 — Reconnect button:**
Click `↺ Reconnect` in the status pill.

**Root causes and fixes:**
| Cause | Fix |
|---|---|
| Browser tab backgrounded on mobile | Keep tab active; add PWA manifest |
| Network timeout (>30s no message) | Add WebSocket keep-alive ping from client |
| Server crashed | Check terminal; restart uvicorn |
| Windows firewall blocked port 8000 | Allow in Windows Defender Firewall |

---

## SOP-014 — Access the App from Another Device

**Requirement:** Host PC and guest device on the same WiFi network.

1. Find host PC IP:
   ```powershell
   ipconfig | findstr "IPv4"
   ```
2. Start server with `--host 0.0.0.0` (default in all run commands)
3. On guest device: `http://<host-ip>:8000`

**If not reachable:**
- Check Windows Firewall: allow inbound on port 8000
- Check router: ensure AP isolation is disabled

---

## SOP-015 — Generate a Guest PIN for Phone Mic

1. Start a meeting (host clicks Start, waits for ● Live)
2. Click `Guest Mic` button → 4-digit PIN displayed
3. Guest opens `http://<host-ip>:8000/guest` on their phone
4. Guest enters PIN → browser requests microphone permission
5. Guest speaks → transcript appears labeled `[Them]`

**PIN resets every meeting.** Valid only while the meeting is active.

---

## SOP-016 — Run the Knowledge Base

**Prerequisite:** Meeting stopped (POST /meeting/stop triggers KB ingestion).

```python
# Manual ingestion (dev/debug use)
from meetingmind.knowledge_base import KnowledgeBase
kb = KnowledgeBase()
kb.ingest_meeting(meeting_id, transcript_chunks, decisions, coaching_events)
```

**Query via context engine:**
```python
from meetingmind.context_engine import ContextEngine
ctx = ContextEngine()
results = ctx.query(current_text="budget discussion", top_k=5)
```

**ChromaDB data location:** `data/knowledge_base/` (configured in `.env`).

---

## SOP-017 — View Meeting Analytics

```python
import sqlite3
conn = sqlite3.connect("data/analytics/meetings.db")
cur = conn.execute("SELECT * FROM meetings ORDER BY date DESC LIMIT 10")
for row in cur: print(row)
```

Or open the file with DB Browser for SQLite.

---

## SOP-018 — Run a Single Test in Watch Mode

```powershell
# Install pytest-watch if not present
pip install pytest-watch

# Watch a specific file
$env:PYTHONPATH="src"; ptw tests/test_coaching.py
```

---

## SOP-019 — Add a New Test File

1. Create `tests/test_<feature>.py`
2. Import from `meetingmind.<module>` (not relative imports)
3. Use `$env:PYTHONPATH="src"` to ensure imports resolve
4. All test functions must start with `test_`
5. Run in isolation first: `python -m pytest tests/test_<feature>.py -v`
6. Confirm count adds to total: run full suite, verify N+new_tests passing

---

## SOP-020 — Commit a Checkpoint

When a sprint milestone is reached (all tests passing, feature complete):

```powershell
# 1. Run tests — record exact count
$env:PYTHONPATH="src"; python -m pytest tests/ -v

# 2. Update MEMORY.md sprint status line
# 3. Create docs/CHECKPOINT_SPRINTn.md
# 4. Update docs/SOP.md if new procedures added

# 5. Stage and commit
git add src/ tests/ docs/ frontend/ MEMORY.md
git commit -m "Sprint N complete: <feature> — N/N tests"

# 6. Push to private only
git push private main
```

---

## SOP-021 — Recover from a Corrupt ChromaDB

**Symptoms:** `KnowledgeBase init failed` in server logs; `/suggestions` returns 503.

```powershell
# Wipe and rebuild
Remove-Item -Recurse -Force data/knowledge_base
# Restart server — KB will be recreated empty on next meeting stop
```

Historical data will be lost but the server will function normally.
Future: implement KB backup before wipe.

---

## SOP-022 — Environment Variable Reference

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Claude API key for suggestions + analysis |
| `LLM_MODEL` | `claude-sonnet-4-6` | Claude model for suggestions |
| `LLM_PROVIDER` | `anthropic` | LLM backend |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding for ChromaDB |
| `AUDIO_DIR` | `audio` | Output path for any saved audio (future) |
| `TRANSCRIPTS_DIR` | `transcripts` | Output path for saved transcripts |
| `OUTPUTS_DIR` | `outputs` | Analysis output path |
| `KB_DIR` | `data/knowledge_base` | ChromaDB persistent storage |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

Copy `.env.example` to `.env` and fill in real values. Never commit `.env`.

---

## SOP-023 — Handle a PyAudio Error on Windows

**Common errors and fixes:**

| Error | Fix |
|---|---|
| `[Errno -9999] Unanticipated host error` | Windows mic privacy blocked — see SOP-024 |
| `[Errno -9997] Invalid sample rate` | Device doesn't support 16 kHz — pick different device index |
| `[Errno -9988] Stream closed` | PyAudio terminated prematurely — restart server |
| No devices found | Check device index with `/devices` endpoint |

---

## SOP-024 — Fix Windows Microphone Privacy Block

**Symptom:** `/health` returns `"windows_mic_blocked": true`; Start fails immediately.

**Fix:**
1. Windows Settings → Privacy & Security → Microphone
2. Enable: "Allow apps to access your microphone"
3. Enable: "Allow desktop apps to access your microphone"
4. Restart the uvicorn server (PyAudio re-checks on startup)

---

## SOP-025 — Enable System Audio Capture (Stereo Mix)

**Requirement:** Capture audio from Zoom/Teams/Meet playing through speakers.

1. Right-click speaker icon in taskbar → Sound settings → More sound settings
2. Recording tab → right-click empty area → Show Disabled Devices
3. Right-click "Stereo Mix" → Enable
4. Note the device index from `/devices`
5. Pass as `system_device_index` to `/meeting/start`

**Alternative:** Install VB-Audio Cable (virtual cable, more reliable than Stereo Mix).

---

## SOP-026 — Update Public Repo After a Sprint

**When:** A sprint is complete and you want to update the public repo with non-proprietary changes.

```powershell
# 1. Identify which files are safe to update publicly
#    Safe: audio_capture.py, transcriber.py, main.py (stripped), index.html (stripped)
#    Never: suggestion_engine.py, context_engine.py, knowledge_base.py, guest_session.py,
#           device_detector.py, token_budget.py, _api_key.py, analyser.py

# 2. Switch to the public branch
git checkout public-strip

# 3. Cherry-pick ONLY safe file changes (do NOT git merge from main)
# Manually copy/edit the relevant parts

# 4. Verify no proprietary references
grep -r "suggestion_engine\|context_engine\|knowledge_base\|ANTHROPIC\|chromadb" src/ frontend/

# 5. Commit and push to origin
git add <safe files>
git commit -m "public: update core transcription"
git push origin public-strip:main

# 6. Return to main
git checkout main
```

---

## SOP-027 — Restore Private Repo to a New Machine

```powershell
# Clone private repo
git clone https://github.com/andrearecheta-alpha/MeetingMind-Private.git MeetingMind
cd MeetingMind

# Add public remote for reference (never pull from it)
git remote add origin https://github.com/andrearecheta-alpha/MeetingMind.git

# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY

# Install ffmpeg (required for Whisper)
winget install ffmpeg

# Run tests
$env:PYTHONPATH="src"; python -m pytest tests/ -v

# Start server
$env:PYTHONPATH="src"; python -m uvicorn meetingmind.main:app --host 0.0.0.0 --port 8000
```

---

## SOP-028 — Branching and Version Control Rules

```
main branch
├── All active development
├── Push to: private only
├── Never pull from: origin (public)
└── Test count must be maintained or increased on every commit

public-strip branch
├── Stripped public version (no proprietary modules)
├── Push to: origin/main only
├── Updated manually — never auto-synced from main
└── Contains: audio_capture.py, transcriber.py, main.py (stripped), index.html (stripped)

Remotes:
├── private  → MeetingMind-Private (all code, all tests)
└── origin   → MeetingMind (public, stripped)
```

**Golden rule:** If you can see `suggestion_engine.py` in your working tree,
you are on `main`. All Sprint 5+ development happens here. Never force-push `main`
to `origin`.

---

## SOP-029 — Push Safety Rule

**CRITICAL: All pushes must go to `private`, never `origin`.**

`origin` is the **public** stripped repo. Pushing proprietary code there exposes
coaching logic, decision detection, suggestion engine, guest session, knowledge
base, and context engine to the public internet.

### Before Every Push

```bash
# 1. ALWAYS verify remotes first
git remote -v
# Confirm you are pushing to "private" (MeetingMind-Private), NOT "origin"
```

### Correct Command

```bash
git push private main
```

### NEVER Use

```bash
git push                 # defaults to origin (PUBLIC)
git push origin main     # pushes to PUBLIC repo
git push --force origin  # overwrites PUBLIC repo history
```

### Incident Response

If proprietary code is accidentally pushed to `origin`:

```bash
# 1. Immediately restore the stripped public commit
git push origin 89866da:main --force

# 2. Verify cleanup
git ls-tree --name-only -r origin/main | grep -E "suggestion_engine|coaching|decision"
# Must return empty

# 3. Contact GitHub support to purge dangling commits if repo is public
```

### Pre-Push Checklist

- [ ] `git remote -v` — confirmed target is `private`
- [ ] No `.env` or credentials in staged files
- [ ] `python -m pytest tests/ -q` — all tests pass

---

## SOP-030 — Stakeholder Privacy

**When:** Writing any code, documentation, test data, KB seeds, or commit messages.

**Rule:** Never use real names of stakeholders, beta users, or clients in:

- Code comments
- Documentation (README, checkpoints, SOPs)
- KB seed files
- Test data and fixtures
- Commit messages
- Action items or meeting notes

**Use instead:** Initials or codenames.

| Real reference | Allowed form |
|----------------|-------------|
| Executive persona | HH |
| Stakeholder names | Initials (e.g. AR, SL) |
| Client companies | Codenames or "Client A" |
| Beta testers | "Tester 1", initials |

**Audit:** Before every commit, verify no real names appear in changed files. Run:

```bash
# Scan for common real-name patterns in tracked files
git diff --cached --name-only | xargs grep -iE "(first last|full name patterns)" || echo "Clean"
```

**Violation response:** Replace immediately with initials, amend or create new commit.

---

## SOP-031 — NLP Upgrade Path

**When:** Adding or modifying any text-matching logic (coaching triggers, decision detection, action items, risk detection).

**Rule:** Follow the 3-stage NLP maturity ladder:

| Stage | Technique | When to Use | Sprint |
|-------|-----------|-------------|--------|
| 1 | Substring/regex patterns | MVP, proof of concept | 1-5 |
| 2 | spaCy dependency parsing + NER | Structured extraction, catches paraphrases | 6-7 |
| 3 | HuggingFace zero-shot classification | Production accuracy, language-agnostic | 8+ |

**Guidelines:**

1. **Never skip stages.** Stage 1 validates the feature concept. Stage 2 proves the NLP approach. Stage 3 replaces both.
2. **Keep old patterns as tests.** When upgrading from Stage 1 to Stage 2, convert the old substring patterns into test assertions. The new NLP function must pass all existing pattern tests PLUS new paraphrase tests.
3. **Dual-run before cutover.** Run old and new detection in parallel for at least one sprint. Log disagreements. Only cut over when the new system matches or exceeds old accuracy.
4. **Error boundaries always.** Every NLP function must have a try/except that returns an empty result on failure. NLP errors must never crash the transcription loop.
5. **Performance budget.** spaCy: < 5ms per chunk. Zero-shot: < 100ms per chunk. If slower, batch or run async.

**Functions per stage:**

| Function | Stage 1 (patterns) | Stage 2 (spaCy) | Stage 3 (zero-shot) |
|----------|-------------------|-----------------|---------------------|
| Action items | `extract_action_items()` | `detect_obligations()` | AI-025 |
| Decisions | `detect_decision()` | `detect_decisions()` | AI-025 |
| Risks | pattern in coaching | `detect_risks()` | AI-025 |
| Coaching | `detect_coaching()` | dep-parse upgrade | AI-025 |
