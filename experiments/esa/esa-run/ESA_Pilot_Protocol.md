# Independent ESA telemetry pilot
**Protocol ESA-M1-H1, version 1.0 · 19 November 2026 · Scientific lead: Astra**

## Question and scope

Does a channel-specific one-step LSTM recover more annotated anomaly events than persistence at an agreed workload on a different spacecraft dataset? The primary comparison is LSTM versus persistence. Level is a secondary comparator. The single-mission pilot uses new ESA observations and freshly fitted models. It evaluates transfer of the NASA study's approach, with a coarser time grid, a common alert-duration policy and labelled calibration. This does not reproduce the official ESA leaderboard. The numerical budgets below are research assumptions pending operator review.

## Data and chronology

Use the ESA Anomaly Dataset version 2, Mission1 only. Take precisely the six published lightweight target channels, channel_41 through channel_46, from ESA-Mission1.zip, plus channels.csv, labels.csv and anomaly_types.csv from the same version. Pin the archive to the release's MD5 9770ad12ed730238f37c42d5c27ab436 and record SHA-256 for the archive and inputs. An extracted subset requires traceability to that archive through a verified extraction manifest or publisher-authenticated member digests before being called verified. A synthetic fixture can validate software only. Keep test annotations out of all configuration and model selection. An incidental literature-search snippet reported aggregate label coverage for the published lightweight subset. No event rows, pilot-period telemetry, held-out forecasts or pilot detector scores were consulted to define this design.

The authors' Mission1 84-month split assigns the boundary at 2007-01-01 00:00 to training. Use the following chronological pilot windows. All times are release-clock timestamps treated as UTC for consistent indexing. The anonymized mission clock has no operational calendar interpretation.

| Block | UTC time interval | Use |
|---|---|---|
| Fit | 2005-10-01 00:00 inclusive to 2006-07-01 00:00 exclusive | Model weights and normalisation |
| Checkpoint selection | 2006-07-01 00:00 inclusive to 2006-10-01 00:00 exclusive | Minimum validation MSE and per-channel reference thresholds |
| Labelled calibration | 2006-10-01 00:00 inclusive through 2007-01-01 00:00 inclusive | Joint workload constraints and threshold choice |
| Evaluation | After 2007-01-01 00:00 through 2008-01-01 00:00 inclusive | One replay after model and threshold freeze |

## Input and causal replay

Use only the target channel as input to each of its three models. Verify monotonic timestamps and numeric values. Collapse exact duplicate timestamps only if all values agree, otherwise fail intake and log the conflicting channel. Align to the top of each hour using the last observation at or before that time, allowing age up to and including two hours. A newer nonfinite observation marks the grid value unavailable. Never backfill. The first fit observation starts at the declared fit boundary; later blocks may use preceding observations within the selected input history.

Score each detector on identical timestamps, requiring a finite target and 250 consecutive available hourly history values. Reject windows crossing a missing grid value. Fit each channel separately. Centre and scale using finite fit-block grid values only (population standard deviation floored at 1e−8). No retrospective labels enter input production, fitting, checkpoint selection or alarm-state updates. Report eligible windows and exclusions separately by channel and block. If a target lacks any usable fit window or checkpoint target, or fewer than 100 scoreable targets in either checkpoint selection or calibration, exclude that channel for all methods and record the reason. With fewer than six retained channels, report partial coverage and withhold the planned six-channel feasibility conclusion.

Use the direct level deviation and absolute persistence error. The LSTM uses 250 preceding standardised target readings, two layers of 80 units, interlayer dropout 0.30, one scalar output, seed 42 per channel, deterministic CPU execution, Adam 0.001, batch size 64, at most 50 epochs and patience five. Minimise MSE, retain the earliest checkpoint attaining the smallest selection error, and score the absolute one-step residual. Apply identical trailing EWMA (current weight 2/31) separately for each method. Reset smoothing and alert state at the first eligible time within each block and after a gap in score eligibility. Save raw scores, indices, eligibility and numerical transformations.

## Thresholds and workload

For each detector and channel, compute a base threshold from the smoothed checkpoint-selection scores, using the observed value at sorted one-based rank ceil(0.99n). Use strict exceedance. Choose one multiplier per detector across its six channels, on the grid 10^(k/40) for integer k from −40 through 40 inclusive. Include a distinct no-alarm endpoint of positive infinity. For a zero base, all finite multipliers give zero. A zero threshold still uses strict exceedance.

Emit at most 30 consecutive flagged readings. Emitting the thirtieth enters lockout. A locked stream rearms only after five consecutive readings at or below threshold, restarting its clear count on each exceedance. A shorter alarm closes at the first clear reading and may rearm immediately. Apply the policy forward in time. Record raw exceedance occupancy, emitted alert occupancy, episode duration and suppressed exceedances. At this hourly cadence the cap is 30 hours. This duration is an experimental transfer assumption.

Apply labelled calibration only after the scores are fixed. A candidate must have at most two unmatched alarm episodes per 1,000 scored channel-hours and at most 5% of scored channel-hours flagged, pooling counts across Mission1. This denominator measures channel monitoring load across six streams. Among feasible finite candidates that actually emit an alert, maximise pooled channel-event recall, then minimise fraction flagged, then false-alarm rate, then choose the larger multiplier. If no such candidate exists or calibration contains no eligible positive event, freeze the no-alarm fallback, report the reason, and withhold evidence of useful detection for that method. Publish all calibration candidates and the selected thresholds before processing evaluation labels. Numerical equality with either workload limit is feasible.

## Reference events and accounting

Join labels to event types by ID, maintaining channel and original timestamp intervals. Category Anomaly supplies positives; Rare Event is nominal for the primary endpoint, with overlapping alarm counts reported separately. Communication Gap and Invalid Data annotations are censored only by the evaluator. Unexpected categories or ambiguous ID mappings are intake discrepancies and require a documented resolution before evaluative claims. Retroactive annotation-based censoring cannot reset the streaming model. Restrict subsequent scoring to eligible, uncensored hourly grid instants. Alarm episodes spanning a censored or missing point split at that point for evaluation. Report the uncensored duration and raw streaming alert metrics separately so censoring choices are auditable.

Define each truth as a unique (channel, event ID) with support equal to the union of its annotated Anomaly intervals on eligible grid instants. Within a channel, count a truth once across any support fragments. Count and exclude labels with no eligible instant, including sub-hour or fully censored intervals. Report the count for each reason. An alarm can match a truth only where both have an eligible instant in common. Choose a maximum-cardinality one-to-one matching; among ties minimise summed delay from each truth's first eligible instant to its first overlap, then prefer earlier truth start, alarm start and stable ID order. Report matches, misses, unmatched alarms, event precision, recall, false alarms per 1,000 scored channel-hours, flagged fraction and delay in hours on matched events. Give denominators and counts by channel and pooled for the mission; label pooled truths as channel-events, since a physical incident may affect several channels.

## Reporting and decision

Seal code and dependency versions, split manifest, source hashes, fitted transforms, checkpoints, calibration results and numerical thresholds before the test evaluation. Save test score and flag hashes before opening the test labels for the evaluator. Compare LSTM versus persistence on exactly the same exposure. The pilot supports a favourable *descriptive* transfer result only if both methods have a nonfallback calibration setting, a positive test event denominator, and satisfy both observed test workload caps, and LSTM detects more channel-events. Otherwise report the observed trade-off, budget failures or missing evidence. If evaluation contains no eligible positive, recall is undefined. Do not retune or extend the test period after seeing outcomes. This single spacecraft, fixed channel family and one year support no population-wide or deployment inference.

## Primary sources

- ESA dataset v2 and published integrity information. [Zenodo](https://zenodo.org/records/15237121)
- Authors' benchmark code, lightweight subsets and preprocessing. [ESA-ADB](https://github.com/kplabs-pl/ESA-ADB)
- Dataset construction, label definitions and official split design. [Research paper](https://openreview.net/pdf?id=FYEGPuUrpo)
