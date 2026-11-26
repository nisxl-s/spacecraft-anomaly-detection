# NASA SMAP/MSL labeled test replay

This is an exploratory comparison. The project's earlier benchmark already used the official NASA test split. The replay compares the direct LSTM, persistence-residual LSTM, and persistence forecast for the same 78 channels with frozen calibration thresholds. No model fitting or threshold adjustment takes place during this run.

The first 250 readings in each test series provide forecast history. Starting at index 250, forecasts use only samples preceding each target. Each arm starts a fresh EWMA and alarm state at the scoring boundary. Alarms require two consecutive strict threshold exceedances, have a 30-reading cap, and rearm after five clear readings following a cap. Predictions are saved and hashed before loading the label CSV for event scoring.

Event matching maximizes the number of one-to-one overlaps between alarm and labeled event intervals, then minimizes total delay. An event fully before the scoring range is excluded, and a partially observable event is clipped to the scoring range. Delays are in readings, measured from the observable onset of matched events. The false alarm rate counts unmatched alarm episodes per 1,000 scored readings.

- `summary.json` contains pooled and mission-level results.
- `ledger.json` contains event intervals, matches, alarms, and thresholds by channel.
- `replay-manifest.json` records score file hashes and thresholds fixed before scoring.
- `scores/` contains the smoothed scores and emitted flags indexed to the original test series.
- `run.log` records channel completion.
- `evaluate.py` implements the replay using the existing NASA benchmark code, pinned archive and label CSV, trained direct-model files, and the complete residual LSTM checkpoint archive. It expects the checkpoint archive in `../verified-complete/` relative to the script and the original benchmark and residual screening modules on Python's import path.
