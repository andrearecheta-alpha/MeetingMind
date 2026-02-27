# MeetingMind — Product Vision

**"The AI Chief of Staff in your ear"**

---

## The Problem

Professionals spend 30–50 % of their working week in meetings.
Most of that time is structurally wasteful:

- Decisions made without the right data surface
- Action items assigned but never tracked
- Stakeholder signals missed in real time
- No institutional memory between meetings

Existing tools record and summarise. They do not *think*.
MeetingMind acts — in the moment, before the opportunity is gone.

---

## Ideal Customer Profile

### Primary: Program Managers
Running cross-functional programs with 5–20 stakeholders.
Pain: keeping every workstream aligned while managing upward and sideways simultaneously.
Outcome sought: fewer surprises, faster decisions, audit-ready records.

### Primary: Sales Engineers
On live demos and technical discovery calls with prospects.
Pain: tracking open questions, competitive objections, and commitment signals while talking.
Outcome sought: higher close rates, shorter sales cycles, consistent follow-through.

### Primary: C-Suite Executives
Back-to-back schedules, every conversation a decision point or a relationship investment.
Pain: no time to review notes; commitments slip; staff updates are filtered by the time they arrive.
Outcome sought: nothing falls through; their word is always kept; trends surface before they become crises.

### Secondary: Consultants and Advisors
Billing by the hour, accountable to multiple clients simultaneously.
Pain: context-switching overhead; client expectations drift between engagements.
Outcome sought: defensible records, faster ramp-up after gaps, proof of value delivered.

---

## Four-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4 — PROOF                                            │
│  Searchable institutional memory. Every decision, every     │
│  commitment, every outcome — timestamped and retrievable.   │
│  Feeds compliance, performance reviews, client reporting.   │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 — COACHING                                         │
│  Real-time in-ear guidance. Detects patterns, flags risks,  │
│  suggests language, prompts follow-up. The AI Chief of      │
│  Staff whispering what your best advisor would say.         │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — INTELLIGENCE                                      │
│  Continuous analysis of the live transcript. Extracts       │
│  decisions, action items, risks, sentiment, stakeholder     │
│  signals, and meeting quality in real time.                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — AWARENESS                                        │
│  Device-agnostic audio capture. Microphone, system audio,   │
│  phone call, earbuds. Speaker diarisation. 40+ languages.   │
│  Runs entirely on-device — no raw audio leaves the machine. │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1 — Awareness
Local Whisper transcription with speaker diarisation.
Audio sources: laptop mic, system loopback, Bluetooth earbuds, phone (companion app).
Privacy guarantee: raw audio is never transmitted. Only transcript text moves downstream.

### Layer 2 — Intelligence
Continuous Claude analysis of the rolling transcript window.
Extracts structured signals every 30–60 seconds:
- Decisions made (with owner and date)
- Action items (with assignee and deadline)
- Open questions and blockers
- Stakeholder sentiment shifts
- Risk indicators (scope creep language, commitment hedging, budget pressure signals)

### Layer 3 — Coaching
Real-time suggestions delivered via earbuds or a discreet on-screen overlay.
Triggered by conversation events detected in Layer 2.
Examples: prompt the PM to assign an owner before the meeting ends;
alert the SE that a competitor was just mentioned; remind the exec of a prior commitment.
See `COACHING_PROMPTS.md` for the full trigger library.

### Layer 4 — Proof
Persistent, searchable knowledge base (ChromaDB).
Every meeting produces a structured record: summary, decisions, action items, verbatim quotes.
Cross-meeting queries: "What did we commit to Acme last quarter?" answered in seconds.
Export: Markdown, PDF, JSON. Integrations: Jira, Notion, Salesforce (roadmap).

---

## Pricing

| Tier | Price | Included |
|---|---|---|
| **Professional** | $99 / month | 1 user · unlimited meetings · all 4 layers · 90-day knowledge base · email export |
| **Team** | $299 / month | Up to 5 users · shared knowledge base · team analytics dashboard · Slack integration · priority support |
| **Enterprise** | Custom | Unlimited users · SSO · on-premise deployment option · custom integrations · SLA · dedicated CSM |

Annual billing: 2 months free (equivalent to ~17 % discount).
Pilot program: 14-day full-feature trial, no credit card required.

---

## Device Strategy

### Principle: meet users where they already are

Users will not change their hardware for a productivity tool.
MeetingMind must work on the devices already in the room.

### Phase 1 — Phone + Laptop (now)
- **Laptop**: local Whisper capture via mic and WASAPI loopback (Windows); CoreAudio (macOS).
- **Phone companion app**: captures in-person and phone-call audio; streams transcript to laptop over local network. No cloud relay required.
- **Earbuds**: any Bluetooth earbuds already paired to the phone receive coaching audio.

This setup covers: in-person meetings, video calls (Zoom/Teams/Meet), phone calls, hybrid rooms.

### Phase 2 — Smart Glasses (2026–2027)
Audio-capable glasses (Ray-Ban Meta, future form factors) as a standalone capture and coaching device.
Coaching delivered as visual overlay rather than audio — zero disruption to conversation flow.
Laptop becomes optional; phone remains the processing hub.

### Phase 3 — Ambient Room Intelligence (2027+)
Always-on room devices with wake-word activation.
Meeting starts automatically when quorum is detected.
No phone or laptop required for capture.

---

## Competitive Differentiation

| Capability | Otter.ai | Fireflies | Notion AI | MeetingMind |
|---|---|---|---|---|
| Real-time coaching | — | — | — | ✓ |
| On-device audio (privacy) | — | — | — | ✓ |
| Structured intelligence extraction | partial | partial | partial | ✓ |
| Cross-meeting knowledge base | — | — | ✓ | ✓ |
| Phone + laptop capture | — | — | — | ✓ |
| PMP / governance alerts | — | — | — | ✓ |

MeetingMind is not a better note-taker. It is the first tool that acts as a participant.

---

## North Star Metric

**Decisions actioned within 48 hours / decisions made** — the follow-through rate.

If this number improves, everything else improves: trust, velocity, outcomes.
MeetingMind owns this metric. All features are justified by their contribution to it.
