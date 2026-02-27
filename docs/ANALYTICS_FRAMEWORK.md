# MeetingMind — Analytics Framework

Enterprise metrics that turn meeting data into organisational intelligence.
All scores are computed from structured data produced by the Intelligence layer —
never from subjective self-reporting.

---

## 1. Meeting Quality Score (MQS)

A single 0–100 score that summarises how well a meeting converted time into outcomes.
Computed at meeting end; available in real time as a live estimate during the meeting.

### Formula

```
MQS = (
    W_decision  × DecisionScore   +
    W_action    × ActionScore     +
    W_inclusion × InclusionScore  +
    W_time      × TimeScore
) × 100
```

Default weights (sum to 1.0):

| Component | Weight | Rationale |
|---|---|---|
| DecisionScore | 0.35 | Decisions are the primary output of most meetings |
| ActionScore | 0.25 | Uncommitted decisions decay; actions are the commitment mechanism |
| InclusionScore | 0.20 | Unheard voices are a leading indicator of missed risk |
| TimeScore | 0.20 | Overruns and agenda drift destroy downstream schedules |

Weights are configurable per team or meeting type.

### Component Formulas

**DecisionScore**
```
decisions_with_owner    = count of decisions with a named owner
decisions_with_deadline = count of decisions with a deadline
decisions_total         = count of all decisions logged

DecisionScore = (
    0.5 × (decisions_with_owner    / max(decisions_total, 1)) +
    0.5 × (decisions_with_deadline / max(decisions_total, 1))
)

If decisions_total == 0:
    DecisionScore = 0.50  (neutral — absence of decisions is ambiguous)
```

**ActionScore**
```
actions_assigned = count of action items with a named assignee
actions_dated    = count of action items with a deadline
actions_total    = count of all action items

ActionScore = (
    0.5 × (actions_assigned / max(actions_total, 1)) +
    0.5 × (actions_dated    / max(actions_total, 1))
)

If actions_total == 0 AND decisions_total == 0:
    ActionScore = 0.40  (penalty — meeting with no outputs)
```

**InclusionScore**
```
attendees_who_spoke = count of distinct speakers (diarisation)
attendees_total     = count of named attendees on invite

talk_time_gini = Gini coefficient of speaker talk-time distribution
                 (0.0 = perfectly equal; 1.0 = one person spoke entirely)

InclusionScore = (attendees_who_spoke / max(attendees_total, 1)) × (1 - talk_time_gini)
```

**TimeScore**
```
scheduled_duration_s = meeting duration as booked
actual_duration_s    = time from start to stop

overrun_ratio = max(0, (actual_duration_s - scheduled_duration_s) / scheduled_duration_s)

agenda_coverage = items_discussed / max(agenda_items_total, 1)
                  (1.0 if no agenda was loaded)

TimeScore = max(0, 1 - overrun_ratio) × agenda_coverage
```

### MQS Bands

| Score | Band | Interpretation |
|---|---|---|
| 85–100 | Excellent | Clear decisions, accountable owners, everyone engaged, on time |
| 70–84 | Good | Minor gaps — missing dates or one dominant speaker |
| 50–69 | Fair | Significant issues — several unowned decisions or major overrun |
| 30–49 | Poor | Most decisions undecided, low inclusion, or substantial drift |
| 0–29 | Critical | No outputs, one-person monologue, or severe structural problems |

---

## 2. Decision Density

Measures how productively a meeting converts elapsed time into committed decisions.
Tracks organisational decision velocity over time.

### Formula

```
raw_density = decisions_total / (actual_duration_minutes)

# Normalise to per-hour for comparability across meeting lengths
decision_density = raw_density × 60   # decisions per hour
```

### Quality-Weighted Variant

Unowned or undated decisions inflate raw density without creating real outcomes.
The quality-weighted variant discounts low-quality decisions:

```
quality_weight(d) =
    1.00  if owner present AND deadline present
    0.60  if owner present, deadline missing
    0.40  if owner missing, deadline present
    0.15  if owner missing AND deadline missing

weighted_decisions = sum(quality_weight(d) for d in decisions)
decision_density_weighted = (weighted_decisions / actual_duration_minutes) × 60
```

### Benchmarks (calibrate against your organisation's baseline)

| Context | Target density (raw) |
|---|---|
| Executive decision meeting | ≥ 3.0 decisions / hour |
| Project governance review | ≥ 2.0 decisions / hour |
| Status update / standup | ≥ 1.0 decisions / hour |
| Brainstorming / discovery | ≥ 0.5 decisions / hour |

---

## 3. Talk Time Ratio

Tracks speaker participation distribution to surface inclusion issues and
identify communication patterns across people and meeting types.

### Per-Meeting Metrics

```
For each speaker s in attendees:
    talk_time_s[s]   = total seconds of speech attributed to s
    talk_share[s]    = talk_time_s[s] / sum(talk_time_s.values())
    word_count[s]    = words spoken by s
    turns[s]         = number of distinct speaking turns by s
    avg_turn_s[s]    = talk_time_s[s] / max(turns[s], 1)
```

### Distribution Metrics

```
# Gini coefficient — inequality of talk time
# Values: 0.0 (equal) → 1.0 (one person spoke all)
gini = sum(
    abs(talk_share[i] - talk_share[j])
    for i in speakers for j in speakers
) / (2 × n² × mean(talk_share))

# Host-to-group ratio — how much the meeting host dominates
host_ratio = talk_share[host] / max(1 - talk_share[host], 0.01)

# Silent attendee rate
silent_rate = (attendees_who_did_not_speak / attendees_total)
```

### Interpretation Guide

| Metric | Healthy range | Flag if |
|---|---|---|
| Host talk share | 20–40 % | > 60 % |
| Gini coefficient | 0.15–0.45 | > 0.65 |
| Silent attendee rate | < 20 % | > 40 % |
| Avg turn duration | 15–45 s | > 120 s (monologue risk) |

### Trend Alerts (monthly rollup)
- A team whose Gini coefficient has increased for 3 consecutive months is flagged for review.
- An individual whose silent attendee rate exceeds 50 % across meetings triggers a manager nudge.

---

## 4. Follow-Up Compliance

Measures whether action items created in meetings are actually completed.
This is the operational definition of the north star metric (follow-through rate).

### Definitions

```
action_item:   A task with an assignee and a deadline, logged during a meeting.
completion:    The assignee confirms completion (via app check-in or linked integration).
overdue:       Deadline passed without confirmed completion.
```

### Formulas

```
# Completion rate — primary metric
completion_rate = completed_on_time / max(actions_due_in_period, 1)

# On-time rate (subset — completed before deadline)
on_time_rate = completed_before_deadline / max(completed_total, 1)

# Slip rate
slip_rate = overdue_total / max(actions_due_in_period, 1)

# Average days to completion (for completed items only)
avg_completion_days = mean(completion_date - assigned_date for completed items)
```

### Per-Person Metrics

Computed per assignee over a rolling 30-day window:

```
personal_completion_rate[person]  = their completed / their due
personal_on_time_rate[person]     = their on-time / their completed
avg_overdue_days[person]          = mean days overdue for their late items
```

### Per-Meeting-Type Metrics

Track whether certain meeting types systematically produce low follow-through:

```
completion_rate_by_type = {
    "executive_review":     0.87,
    "project_governance":   0.72,
    "status_update":        0.61,
    "brainstorming":        0.44,   ← flag for intervention
}
```

### Compliance Tiers

| Rate | Tier | Action |
|---|---|---|
| ≥ 85 % | Green | No action |
| 70–84 % | Yellow | Monthly trend review |
| 50–69 % | Orange | Manager dashboard alert |
| < 50 % | Red | Escalation + coaching prompt activated |

---

## 5. Monthly Dashboard Metrics

Displayed in the Team and Enterprise dashboards. Computed over a rolling 30-day window
unless otherwise noted.

### Executive Summary Cards

```
Total Meetings          count of meetings in period
Total Meeting Hours     sum(actual_duration_minutes) / 60
Avg Participants        mean(attendees_total per meeting)
Avg MQS                 mean(MQS across all meetings)
Decision Volume         sum(decisions_total)
Decision Velocity       sum(decisions_total) / total_meeting_hours
Action Items Created    sum(actions_total)
Follow-Through Rate     completion_rate (period aggregate)
```

### Trend Charts (time series, weekly granularity)

| Chart | Signal |
|---|---|
| MQS over time | Is meeting quality improving? |
| Decision density | Are we deciding faster? |
| Follow-through rate | Are we delivering on commitments? |
| Talk time Gini | Is inclusion improving? |
| Meeting hours per person | Is meeting load sustainable? |

### Team Leaderboards (opt-in)

Ranked by: MQS (meeting chair), follow-through rate (individual).
Leaderboards are opt-in at the organisation level and hidden by default.
Individual scores are visible only to the individual and their direct manager unless the org
explicitly enables full transparency.

### Anomaly Alerts (email or Slack digest, weekly)

Fired when any metric crosses a threshold relative to the team's own rolling baseline:

```
MQS drops > 15 points week-over-week       → "Meeting quality decline detected"
Follow-through rate < 50 % for any team   → "Action item compliance critical"
Meeting hours/person > 25 h/week          → "Meeting load warning"
Gini > 0.70 for any recurring meeting     → "Inclusion concern flagged"
Silent attendee rate > 50 % recurring     → "Recurring meeting audit recommended"
Decision density < 0.5 for exec meeting   → "Decision velocity low"
```

### Quarterly Business Review Pack

Auto-generated PDF report containing:

1. Executive summary: MQS trend, follow-through rate, decision velocity
2. Top 10 decisions by impact score (user-rated or Claude-inferred)
3. Outstanding action items older than 30 days
4. Meeting type breakdown: hours and MQS by category
5. Top 5 individual contributors by follow-through rate (opt-in)
6. Recommendations: meetings to cancel, frequency to reduce, or format to change

---

## Implementation Notes

### Data Storage
All metrics are derived from structured meeting records in ChromaDB + a relational
summary table (SQLite for self-hosted; Postgres for Enterprise).

Raw transcript text is retained for 90 days (Professional), 1 year (Team), or
as configured (Enterprise). Computed metrics are retained indefinitely.

### Privacy Controls
- Talk time and per-person metrics are computed locally before any sync.
- Individual scores are never included in aggregate API calls to Claude.
- Organisations can disable per-person leaderboards without losing aggregate metrics.
- GDPR right-to-erasure: individual records deleted on request; aggregate metrics
  are anonymised (not deleted) since they no longer contain personal data.

### Calibration Period
MQS benchmarks assume a minimum of 10 meetings per team before trend alerts activate.
This prevents false positives during the first two weeks of use.
