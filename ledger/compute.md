# Compute ledger

Modal billing export: 1083 hourly rows, 291 apps, 2026-10-17T03:00:00 to 2026-10-18T08:00:00 UTC (`billing_export.json`).

| | USD | H100-hours (at $3.95) |
|---|---|---|
| H100, all apps | 783.69 | 198.4 |
| H100, apps matched to a named job | 765.27 | 193.7 |
| H100, unmatched apps (22: manual runs, tests, the peer session) | 18.42 | 4.7 |
| H100, paper phase (jobs started from 2026-10-17 12:20 PDT) | 379.88 | 96.2 |
| CPU, memory and egress, all apps | 28.37 | |
| **total billed** | **812.06** | |

Cross-check for the paper phase, whose GPU counts are known: start to end × GPUs by the queue clock gives 103.3 H100-hours against 96.2 billed. The clock also counts container scheduling and start-up; billing is the authority.

Billing lags by about an hour, so jobs that ended within the last hour are not fully billed yet.
