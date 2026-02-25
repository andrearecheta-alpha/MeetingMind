# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | Yes       |

---

## Data & Privacy Considerations

MeetingMind processes **audio recordings and transcripts that may contain
personally identifiable information (PII), confidential business discussions,
or legally protected content.** All contributors and operators must:

- Store audio and transcript data **locally or in access-controlled, encrypted
  storage only**. Never commit raw audio or transcripts to version control.
- Honour applicable data-protection regulations (GDPR, CCPA, HIPAA, etc.)
  before processing any meeting recordings.
- Obtain explicit consent from all meeting participants before recording or
  processing audio.

---

## Secrets Management

- API keys (OpenAI, Anthropic, AWS, etc.) must be stored in `.env` only.
- `.env` is listed in `.gitignore` and **must never be committed**.
- Use `.env.example` (no real values) to document required variables.
- Rotate any key that is accidentally exposed immediately.

---

## Dependency Security

- Pin dependency versions in `requirements.txt` to reproducible hashes where
  possible (`pip-compile --generate-hashes`).
- Run `pip audit` or `safety check` regularly to surface known CVEs.
- Keep Python and all packages up to date.

---

## Reporting a Vulnerability

If you discover a security vulnerability in MeetingMind, **please do not open
a public GitHub issue.**

Instead, report it privately:

1. Email: `security@<your-domain>.com`  *(update before publishing)*
2. Include a clear description, reproduction steps, and potential impact.
3. Allow a reasonable disclosure window (90 days) before public disclosure.

You will receive an acknowledgement within **48 hours** and a remediation
timeline within **7 days**.

---

## Threat Model (summary)

| Threat | Mitigation |
|---|---|
| Leaked API keys | `.env` in `.gitignore`; secrets scanning in CI |
| PII in audio/transcripts committed to git | Audio & transcript paths excluded via `.gitignore` |
| Malicious audio file (e.g. prompt injection via transcript) | Sanitise all LLM inputs; treat transcripts as untrusted |
| Insecure third-party dependencies | Regular `pip audit`; pinned hashes |
| Unauthorised KB access | File-system permissions; auth layer before any web API |
