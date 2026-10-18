# Compute ledger

Modal billing export: 1075 hourly rows, 290 apps, 2026-10-17T03:00:00 to 2026-10-18T06:00:00 UTC (`billing_export.json`).

| | USD | H100-hours (at $3.95) |
|---|---|---|
| H100, all apps | 722.37 | 182.9 |
| H100, apps matched to a named job | 703.94 | 178.2 |
| H100, unmatched apps (22: manual runs, tests, the peer session) | 18.42 | 4.7 |
| H100, paper phase (jobs started from 2026-10-17 12:20 PDT) | 318.56 | 80.6 |
| CPU, memory and egress, all apps | 26.57 | |
| **total billed** | **748.93** | |

Cross-check for the paper phase, whose GPU counts are known: start to end × GPUs by the queue clock gives 94.5 H100-hours against 80.6 billed. The clock also counts container scheduling and start-up; billing is the authority.

Billing lags by about an hour, so jobs that ended within the last hour are not fully billed yet.
