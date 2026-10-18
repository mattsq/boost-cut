# Compute ledger

Modal billing export: 1053 hourly rows, 286 apps, 2026-10-17T03:00:00 to 2026-10-18T04:00:00 UTC (`billing_export.json`).

| | USD | H100-hours (at $3.95) |
|---|---|---|
| H100, all apps | 661.28 | 167.4 |
| H100, apps matched to a named job | 642.85 | 162.7 |
| H100, unmatched apps (22: manual runs, tests, the peer session) | 18.42 | 4.7 |
| H100, paper phase (jobs started from 2026-10-17 12:20 PDT) | 257.47 | 65.2 |
| CPU, memory and egress, all apps | 24.74 | |
| **total billed** | **686.02** | |

Cross-check for the paper phase, whose GPU counts are known: start to end × GPUs by the queue clock gives 74.5 H100-hours against 65.2 billed. The clock also counts container scheduling and start-up; billing is the authority.

Billing lags by about an hour, so jobs that ended within the last hour are not fully billed yet.
