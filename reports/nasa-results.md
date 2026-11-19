# NASA spacecraft anomaly benchmark results

## Frozen held-out replay

The benchmark is complete across 78 channels, 100 expert-labelled events and 472,832 scored readings. The 52 SMAP channels account for 66 events and 406,699 readings. The 26 Mars Science Laboratory (MSL) channels account for 34 events and 66,133 readings.

| Scope | Detector | Events detected | Unmatched alarm episodes |
| --- | --- | ---: | ---: |
| Overall | Level | 70/100 | 775 |
| Overall | Persistence | 78/100 | 1,401 |
| Overall | Direct LSTM | 90/100 | 2,148 |
| SMAP | Level | 45/66 | 614 |
| SMAP | Persistence | 50/66 | 1,056 |
| SMAP | Direct LSTM | 61/66 | 1,913 |
| MSL | Level | 25/34 | 161 |
| MSL | Persistence | 28/34 | 345 |
| MSL | Direct LSTM | 29/34 | 235 |

The LSTM missed the primary criterion of higher recall without an increase in unmatched alarms overall. On MSL it detected one more event than persistence with 110 fewer unmatched episodes. Unmatched episodes are measured against the supplied annotations. The upstream release used test-set extrema for scaling, and anonymised timestamps limit delay reporting to readings.

## Exploratory alarm-policy replay

Matching detector alarm counts on calibration did not equalise unmatched alarm rates on the held-out set. In the threshold sweep, only zero-alert settings satisfied both proposed workload limits of two unmatched alarms per 1,000 readings and 5% of readings flagged.

Capping alarms at 30 readings and requiring five clear readings to reset after the cap reduced direct-LSTM detections from 90 to 76 and the flagged fraction from 33.51% to 6.71%. On that capped policy, requiring two consecutive exceedances preserved 76 detections and reduced unmatched alarm episodes from 1,976 to 1,682. The flagged fraction fell to 6.44% and median detection delay rose from 25.5 to 26.5 readings. These follow-ups reused the NASA held-out set.

## Persistence-residual forecasting screen

The residual model predicts the change from the previous reading. The screen compares it with the direct LSTM and persistence on chronological partitions of the official NASA training files. The final partition is a nominal evaluation, with thresholds frozen from a separate calibration partition.

The run stopped after 30 of 78 channels, including 12 SMAP and 18 MSL. The verified partial checkpoint covers 4,026 evaluation readings.

| Detector | Mean of channel MAEs | Nominal alarm episodes | Readings flagged |
| --- | ---: | ---: | ---: |
| Direct LSTM | 0.06523 | 30 | 517 |
| Residual LSTM | 0.03583 | 19 | 351 |
| Persistence | 0.03665 | 11 | 198 |

The screening decision requires all 78 channels and a final audit. Nominal alarm counts do not measure labelled-event detection. Independent validation on ESA Mission1 v2 is pending. Synthetic intake and forecasting checks passed on six streams.
