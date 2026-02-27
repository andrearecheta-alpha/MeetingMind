# MeetingMind — Coaching Prompt Library

Real-time suggestions delivered to the user during a live meeting.
Each entry defines: the trigger condition, the text shown/spoken, and the minimum
confidence threshold before the prompt fires.

Confidence is a float in [0.0, 1.0] produced by the Intelligence layer analysis.
Prompts below threshold are queued and re-evaluated on the next analysis cycle.

---

## Category 1 — Stakeholder Management

### 1.1 Unnamed Decision Owner
**Trigger:** A decision is detected in the transcript without an explicit owner assigned.
Detection signal: decision-language phrase + no named individual within 30 words.
```
Trigger condition:   decision_detected AND owner_missing
Confidence threshold: 0.75
Prompt text:
  "Decision logged — no owner yet.
   Ask: 'Who's taking this one?'"
Delivery:            overlay + optional audio nudge
Cooldown:            one per meeting per decision
```

### 1.2 Stakeholder Not Heard
**Trigger:** A named attendee has not spoken in more than 8 minutes in an active discussion.
Detection signal: speaker diarisation silence gap > 8 min for a known attendee.
```
Trigger condition:   attendee_silent_minutes >= 8 AND meeting_elapsed_minutes >= 10
Confidence threshold: 0.90 (diarisation confidence gate)
Prompt text:
  "[Name] hasn't spoken in a while.
   Consider: 'What's your read, [Name]?'"
Delivery:            overlay only (silent)
Cooldown:            12 minutes per attendee
```

### 1.3 Commitment Hedge Detected
**Trigger:** A stakeholder uses hedge language after being asked for a commitment.
Detection signal: phrases such as "I'll try", "probably", "should be able to", "we'll see"
following a direct request.
```
Trigger condition:   commitment_requested AND hedge_phrase_detected
Confidence threshold: 0.70
Prompt text:
  "Soft commitment from [Name].
   Clarify: 'Can we put a date on that?'"
Delivery:            overlay only
Cooldown:            5 minutes per speaker
```

### 1.4 Escalation Signal
**Trigger:** Language indicating frustration, urgency, or an implicit escalation request
from a senior stakeholder.
Detection signal: elevated-sentiment phrases + seniority indicator from speaker profile.
```
Trigger condition:   frustration_sentiment >= 0.65 AND speaker_seniority IN [Director, VP, C-suite]
Confidence threshold: 0.72
Prompt text:
  "Tension signal from [Name].
   Acknowledge before moving on."
Delivery:            overlay only
Cooldown:            10 minutes per speaker
```

### 1.5 Prior Commitment Referenced
**Trigger:** A stakeholder references a past commitment that exists in the knowledge base.
Detection signal: semantic match between transcript phrase and stored decision/action item.
```
Trigger condition:   kb_match_score >= 0.80 AND match_type IN [commitment, decision]
Confidence threshold: 0.80
Prompt text:
  "This matches a prior commitment — [date].
   Knowledge base: [short quote]"
Delivery:            overlay (tap to expand full record)
Cooldown:            none (surface every match)
```

---

## Category 2 — Meeting Effectiveness

### 2.1 Agenda Drift
**Trigger:** Conversation topic has diverged from the stated meeting agenda for more than
3 minutes.
Detection signal: semantic distance between current topic embedding and agenda item embeddings.
```
Trigger condition:   topic_drift_minutes >= 3 AND agenda_loaded == True
Confidence threshold: 0.68
Prompt text:
  "Off-agenda for 3 min.
   Redirect or park: 'Should we add this to the next agenda?'"
Delivery:            overlay
Cooldown:            8 minutes
```

### 2.2 No Decisions After 20 Minutes
**Trigger:** Meeting has been active for 20 minutes with zero decisions logged.
Detection signal: elapsed time > 20 min AND decision_count == 0.
```
Trigger condition:   elapsed_minutes >= 20 AND decisions_logged == 0
Confidence threshold: 1.00 (rule-based, no ML required)
Prompt text:
  "20 min in — no decisions logged yet.
   Is this meeting on track?"
Delivery:            overlay
Cooldown:            15 minutes
```

### 2.3 Meeting Nearing End, Open Items Remain
**Trigger:** Fewer than 5 minutes remain and unresolved action items or open questions exist.
Detection signal: (meeting_end_time - now) < 5 min AND open_items_count > 0.
```
Trigger condition:   minutes_remaining <= 5 AND open_items_count > 0
Confidence threshold: 1.00 (rule-based)
Prompt text:
  "[N] open items — [M] minutes left.
   Rapid round: owner + date for each?"
Delivery:            overlay + audio nudge
Cooldown:            2 minutes
```

### 2.4 Monologue Detected
**Trigger:** The user (or another speaker) has spoken uninterrupted for more than 90 seconds.
Detection signal: continuous speaker segment > 90 s.
```
Trigger condition:   continuous_speaker_seconds >= 90
Confidence threshold: 0.95 (diarisation confidence gate)
Prompt text (user is speaker):
  "You've been speaking for 90 sec.
   Check in: invite a reaction."
Prompt text (other speaker):
  "[Name] monologue — 90 sec.
   Opportunity to redirect or interject."
Delivery:            overlay only
Cooldown:            3 minutes per speaker
```

### 2.5 Repeated Question
**Trigger:** The same question (semantically) has been asked more than once without resolution.
Detection signal: high cosine similarity between two question-classified transcript segments.
```
Trigger condition:   question_similarity >= 0.85 AND prior_question_unresolved == True
Confidence threshold: 0.73
Prompt text:
  "This question came up before — still unresolved.
   Flag or assign: 'Let's make sure we close this.'"
Delivery:            overlay
Cooldown:            none per unique question pair
```

---

## Category 3 — Decision Quality

### 3.1 Missing Success Criteria
**Trigger:** A decision is logged but no measurable success criteria or metric is mentioned.
Detection signal: decision detected AND no numeric / measurable phrase within context window.
```
Trigger condition:   decision_detected AND success_metric_missing == True
Confidence threshold: 0.72
Prompt text:
  "Decision logged — no success metric.
   Ask: 'How will we know this worked?'"
Delivery:            overlay
Cooldown:            one per decision
```

### 3.2 No Deadline Assigned
**Trigger:** An action item is assigned to a person but no deadline is stated.
Detection signal: action_item_detected AND date_entity missing from context.
```
Trigger condition:   action_item_detected AND deadline_missing == True
Confidence threshold: 0.75
Prompt text:
  "Action item — no date.
   Confirm: 'When will this be done?'"
Delivery:            overlay
Cooldown:            one per action item
```

### 3.3 Assumption Stated as Fact
**Trigger:** A speaker states an assumption (using assumption-language markers) in a context
where a decision is being made.
Detection signal: assumption-language phrase + decision context within 60-word window.
```
Trigger condition:   assumption_phrase_detected AND decision_context == True
Confidence threshold: 0.68
Prompt text:
  "Assumption flagged: '[quote]'
   Validate before deciding."
Delivery:            overlay
Cooldown:            5 minutes
```

### 3.4 Scope Creep Language
**Trigger:** Phrases indicating scope expansion are detected mid-meeting.
Detection signal: scope-expansion phrases (e.g. "while we're at it", "and also", "we should
probably also") in a planning or delivery context.
```
Trigger condition:   scope_expansion_phrase AND meeting_type IN [planning, delivery, sprint]
Confidence threshold: 0.65
Prompt text:
  "Scope signal: '[quote]'
   Park or log as change request?"
Delivery:            overlay
Cooldown:            5 minutes
```

### 3.5 Risk Raised Without Mitigation
**Trigger:** A risk is explicitly raised but the conversation moves on without a mitigation
or owner being assigned.
Detection signal: risk-language phrase followed by topic change within 90 seconds.
```
Trigger condition:   risk_phrase_detected AND mitigation_missing AND topic_changed_within_seconds <= 90
Confidence threshold: 0.70
Prompt text:
  "Risk raised — no mitigation logged.
   Assign: 'Who owns this risk?'"
Delivery:            overlay
Cooldown:            one per risk phrase
```

---

## Category 4 — PMP Governance Alerts

*These prompts are active when the user's profile includes PMP/PMI certification
or when meeting type is set to `project_governance`.*

### 4.1 Change Control Bypass
**Trigger:** A change to scope, schedule, or budget is agreed informally without reference
to a change control process.
```
Trigger condition:   change_agreed AND change_control_reference_missing == True
Confidence threshold: 0.73
Prompt text:
  "Change agreed without CCB reference.
   Log as change request before this meeting closes."
Delivery:            overlay
Cooldown:            one per change instance
```

### 4.2 Baseline Not Referenced
**Trigger:** Schedule or budget discussion occurs without reference to an approved baseline.
```
Trigger condition:   schedule_or_budget_discussed AND baseline_reference_missing == True AND elapsed_minutes >= 5
Confidence threshold: 0.68
Prompt text:
  "No baseline referenced in schedule/budget discussion.
   Anchor to approved baseline?"
Delivery:            overlay
Cooldown:            10 minutes
```

### 4.3 Lessons Learned Trigger
**Trigger:** A problem or failure is described that matches a pattern from the lessons learned
knowledge base.
```
Trigger condition:   problem_phrase_detected AND kb_lessons_match_score >= 0.78
Confidence threshold: 0.78
Prompt text:
  "Similar issue in knowledge base — [project name, date].
   Lessons learned: [short quote]"
Delivery:            overlay (tap to expand)
Cooldown:            none per unique match
```

### 4.4 Sponsor Absent from Key Decision
**Trigger:** A decision that falls within the sponsor's authority threshold is being made
without the sponsor present.
```
Trigger condition:   decision_authority_level >= threshold AND sponsor_present == False
Confidence threshold: 0.80
Prompt text:
  "This decision may require sponsor sign-off.
   Flag for async confirmation?"
Delivery:            overlay + audio nudge
Cooldown:            one per decision
```

---

## Category 5 — Time-Based Triggers

These fire on elapsed time regardless of transcript content.
Confidence threshold is 1.00 for all (purely rule-based).

### 5.1 Meeting Start — Context Brief
```
Trigger condition:   elapsed_minutes == 1
Prompt text:
  "Meeting started. [N] attendees, [M] agenda items.
   Prior action items due today: [list or 'none']"
Delivery:            overlay
```

### 5.2 Halfway Check
```
Trigger condition:   elapsed_minutes == meeting_duration_minutes / 2
Prompt text:
  "Halfway point. Decisions so far: [N]. Open items: [M].
   On track?"
Delivery:            overlay
```

### 5.3 5-Minute Warning
```
Trigger condition:   minutes_remaining == 5
Prompt text:
  "5 minutes left. [N] open items.
   Confirm owners and dates before closing."
Delivery:            overlay + audio nudge
```

### 5.4 Overrun Alert
```
Trigger condition:   elapsed_minutes > scheduled_duration_minutes
Prompt text:
  "Meeting is running over by [N] min.
   Wrap up or schedule continuation?"
Delivery:            overlay + audio nudge
Repeat interval:     3 minutes
```

### 5.5 Post-Meeting Summary Available
```
Trigger condition:   meeting_stopped AND summary_generated == True
Prompt text:
  "Summary ready. [N] decisions, [M] action items.
   Send to attendees?"
Delivery:            overlay (tap to preview and share)
```

---

## Implementation Notes

### Trigger Evaluation Cycle
The Intelligence layer runs a Claude analysis pass every 30 seconds on the rolling transcript
window (last 90 seconds of speech). Each pass returns a structured JSON payload:

```json
{
  "decisions":    [...],
  "action_items": [...],
  "risks":        [...],
  "sentiment":    {...},
  "signals":      ["commitment_hedge", "scope_expansion"],
  "speakers":     {"Alice": {"words": 340, "last_spoke_s": 12}},
  "confidence":   0.84
}
```

The coaching engine evaluates all trigger conditions against this payload
and fires prompts whose threshold is met.

### Delivery Modes
| Mode | When | User action required |
|---|---|---|
| Overlay (silent) | Default | Dismiss with tap or auto-dismiss in 8 s |
| Audio nudge | High-priority alerts | None |
| Persistent card | Action items, risks | Explicit dismiss |
| Tap to expand | Knowledge base matches | Tap to view full record |

### User Control
- Per-category enable/disable in settings
- Sensitivity slider (maps to global confidence threshold offset: ±0.10)
- Quiet mode: overlay only, no audio, for sensitive conversations
- Do-not-disturb: suppress all prompts for N minutes
