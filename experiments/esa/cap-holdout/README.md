# Search for a cap-stressing spacecraft holdout

The seven-day investigation-record limit and the hourly numeric persistence scorer were held fixed. Alarm arrays were sealed before matching them to the official event intervals.

| Candidate | Evidence |
|---|---|
| ESA Mission 1 channels 61–63 | Three eligible labeled channel-events from one fresh incident. The unchanged scorer produced no confirmed records or pages, so none of the three events overlapped. |
| ESA Mission 2 channel 25, Jan–Oct 2001 | The fresh annotation contained no eligible hourly sample. The longest record was three hours. All 15 original page timestamps were preserved. |
| ESA Mission 3 | The release marks all 13 channels with anomaly labels as categorical targets. The frozen numeric scorer was excluded for this source. |
| OPS-SAT-AD | The official segments.csv matched Zenodo MD5 1f4b6ffd0359efac96a6698ff557e0f9. Its 2,123 published channel segments span at most 22 min 15 s each. Concatenating segments would invent continuity. |

Both replays fell short of the original 30-hour threshold, so they cannot establish an independent coverage benefit for the seven-day rule. No derived anomaly labels were used.
Sources: ESA Anomaly Dataset v2 (https://zenodo.org/records/15237121) and OPS-SAT-AD (https://zenodo.org/records/12588359).

To reproduce, place the respective official telemetry ZIPs, labels.csv and anomaly_types.csv in inputs/ for Mission 1 and m2/inputs/ for Mission 2. Run `python replay.py freeze` then `python replay.py score` from this directory, or pass the same arguments to m2/replay.py for Mission 2.
