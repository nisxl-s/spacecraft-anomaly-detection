# Reproduce the NASA follow-up sequence

Start with the [frozen benchmark](../reports/nasa-results.md) and a Python environment with NumPy, CPU PyTorch 2.8.0, and Matplotlib. The commands below assume the verified NASA archive, byte-identical pinned intake manifest, direct checkpoints with their training reports, P-1 freeze record, and the frozen replay directory are available locally. The replay directory contains the ledger and per-channel scores. The intake command in the repository README regenerates the pinned manifest as `intake-replay.json`.

```sh
python scripts/alarm_rate_comparison.py calibrate --archive kaggle.zip --manifest intake-replay.json --models models --pilot p1-verified.json --output alarm-rate
python scripts/alarm_rate_comparison.py evaluate --replay full_benchmark --output alarm-rate
python scripts/threshold_sweep.py --replay full_benchmark --output threshold-sweep
python scripts/duration_reset.py --replay full_benchmark --sweep-comparisons threshold-sweep/reference_comparisons.json --output duration-reset
python scripts/confirm_alarm.py --replay full_benchmark --previous-policy duration-reset --output two-reading
```

The held-out NASA replays after the frozen benchmark are exploratory. ESA Mission1 v2 is reserved for an independent evaluation. Synthetic checks passed; the real-data pilot remains pending.

## NASA persistence-residual LSTM screening protocol

Compare the original direct LSTM, a persistence-residual LSTM that predicts the change from the previous reading, and a persistence forecast on the 78 previously eligible SMAP and MSL channels. Use only the official NASA training arrays. Reserve ESA for independent validation.

For a channel with N rows, use target boundaries floor(0.70N), floor(0.85N), and floor(0.925N). The sections are model fitting, early stopping, threshold calibration, and nominal evaluation. The first eligible target follows 250 observed rows. Each target uses the same strictly preceding observed window for the two model forecasts. Keep the archive scale and original optimizer, architecture, seed and early stopping budget. Reuse the matching frozen direct-model checkpoint and recompute all thresholds on the new calibration segment.

Smooth the absolute errors with the original EWMA. Use the nearest-rank 99th percentile of the calibration scores and a strict exceedance. Reset the alarm state at the evaluation boundary. Require two consecutive exceedances, cap each episode at 30 flagged readings, and rearm after five clear readings. Retain per-channel inputs, thresholds, forecast and alarm summaries and aggregate by mission and across both missions.

Advance the residual model only if its macro channel MAE is strictly lower than the direct model and no higher than persistence, and its pooled nominal alarm episodes per 1,000 readings and fraction of readings flagged do not exceed the direct model. A partial run cannot establish that the candidate passes. Nominal alarm counts do not estimate event detection, and any later reuse of the official NASA test set is exploratory.

### Resume the residual screen

The [checkpoint summary](../reports/residual-screen-checkpoint.json) records the 30 audited channels. To resume, extract the saved residual checkpoint and point both commands at its `results` directory. Use a single coordinator for that output directory.

```sh
python scripts/audit_residual_screen.py residual-checkpoint/results
python scripts/residual_screen.py --archive kaggle.zip --manifest intake-replay.json --models models --output residual-checkpoint/results --workers 7 --threads 1
python scripts/audit_residual_screen.py residual-checkpoint/results
```

The runner reuses completed channel records. It writes a cohort summary after it finishes. Keep the model weights, per-channel scores and telemetry archives outside Git.
