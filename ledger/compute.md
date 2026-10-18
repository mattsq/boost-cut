# Compute ledger

Modal billing export: 1029 hourly rows, 280 apps, 2026-10-17T03:00:00 to 2026-10-18T03:00:00 UTC (`billing_export.json`).

| | USD | H100-hours (at $3.95) |
|---|---|---|
| H100, all apps | 631.80 | 159.9 |
| H100, apps matched to a named job | 613.38 | 155.3 |
| H100, unmatched apps (22: manual runs, tests, the peer session) | 18.42 | 4.7 |
| H100, paper phase (jobs started from 2026-10-17 12:20 PDT) | 227.99 | 57.7 |
| CPU, memory and egress, all apps | 23.83 | |
| **total billed** | **655.63** | |

Cross-check for the paper phase, whose GPU counts are known: start to end × GPUs by the queue clock gives 65.0 H100-hours against 57.7 billed. The clock also counts container scheduling and start-up; billing is the authority.

Billing lags by about an hour, so jobs that ended within the last hour are not fully billed yet.
