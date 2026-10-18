# Compute ledger

Modal billing export: 1125 hourly rows, 297 apps, 2026-10-17T03:00:00 to 2026-10-18T13:00:00 UTC (`billing_export.json`).

| | USD | H100-hours (at $3.95) |
|---|---|---|
| H100, all apps | 937.30 | 237.3 |
| H100, apps matched to a named job | 918.88 | 232.6 |
| H100, unmatched apps (22: manual runs, tests, the peer session) | 18.42 | 4.7 |
| H100, paper phase (jobs started from 2026-10-17 12:20 PDT) | 533.50 | 135.1 |
| CPU, memory and egress, all apps | 32.92 | |
| **total billed** | **970.22** | |

Cross-check for the paper phase, whose GPU counts are known: start to end × GPUs by the queue clock gives 142.3 H100-hours against 135.1 billed. The clock also counts container scheduling and start-up; billing is the authority.

Billing lags by about an hour, so jobs that ended within the last hour are not fully billed yet.
