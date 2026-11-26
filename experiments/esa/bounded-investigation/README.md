# Seven-day anomaly investigation record

The experiment clips each continuous sustained investigation record to 168 hours. A fresh record requires the signal to clear and confirm again. The paging rule is the original two-high, 30-reading cap and five-low rearm rule. The 168-hour limit is a provisional review horizon, not a duration specified by operators. Both panels use hourly persistence scores rather than the fitted NASA model.

## Reproduction

Run with Python 3.12, NumPy 2 and pandas 2.2.

```sh
python bounded-investigation/verify_controls.py
python bounded-investigation/run.py nasa-independent-validation/followup
python bounded-investigation/mission1/replay.py freeze
python bounded-investigation/mission1/replay.py score
```

The development input is the three-channel sealed panel from Spacecraft_Sustained_Record_Validation.zip. The independent test uses the ESA Anomaly Dataset v2 Mission 1 channel ZIPs 15, 21 and 47, and the official labels.csv and anomaly_types.csv files. Obtain them from https://zenodo.org/records/15237121 and place them under mission1/inputs. The release channel members were verified against the archive's CRC before deserializing trusted pickle tables. The freeze command runs without reading labels. Each per-channel alert array is hashed in the seal before scoring. The package includes frozen arrays, the event ledger and summary. Both runs use event overlap, and in-event pages require a page timestamp inside the event interval.

## Results

| Panel | Labeled channel-events covered | Record occupancy | Original page timestamps |
|---|---|---:|---:|
| Mission 2 development | capped 2/5, uncapped 3/5, seven-day 3/5 | capped 0.76%, uncapped 11.30%, seven-day 1.49% | all 142 preserved |
| Mission 1 untouched incidents | all three variants 1/26 | all three variants 0.42% | all 714 preserved |

Mission 1 included 26 labeled channel-events across 25 distinct incidents after excluding incidents linked to previously evaluated channels. The longest record was nine hours, below the original 30-hour cap. This verifies paging parity and eligibility on unseen incidents but cannot establish that the seven-day rule retains extra coverage out of sample. The low event coverage also limits the hourly persistence scorer as a validation fixture. The seven-day rule remains an investigation-only candidate.
