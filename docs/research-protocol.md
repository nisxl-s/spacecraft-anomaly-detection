# Spacecraft anomaly detection
**Project charter and research protocol · Version 1.0 · 19 November 2026**

**Decision.** Use archived SMAP satellite and Mars Science Laboratory (MSL) rover telemetry to test whether a one-step forecasting model can detect more expert-labelled events than simple alarms at a comparable false-alarm burden. The intended reader is an engineer deciding whether the extra modelling effort is justified. Project selection is approved. This edition defines the experiment before implementation.

## 1. Scope and success

Start with SMAP power channel P-1. It has 8,505 test readings and three labelled contextual events. A contextual anomaly can look ordinary in isolation while being unusual given recent behaviour. P-1 is a pipeline check. Its three events provide too little evidence for a mission-wide performance claim. Reserve its official test labels for scoring under the same frozen rules used on all channels. Use synthetic examples and training-file replay for pilot debugging. [2, 3]

The full study targets both missions. The release reports 55 SMAP and 27 MSL channels and 105 labelled intervals. Treat those counts as reconciliation targets and publish the counts actually recovered, scored and excluded. Deliverables after implementation are a versioned manifest, replayable detector, event ledger, mission-level results table and annotated replay plots. Live feeds, spacecraft commands, physical fault diagnosis and operational certification are outside this study. [2]

**Primary hypothesis.** Across the eligible channels, the LSTM will achieve higher pooled event recall than each baseline with no increase in unmatched alarm episodes per 1,000 scored readings. Evaluate the two conditions jointly at the prespecified operating point. If either fails, report the trade-off and decline the superiority claim. A reproducible negative result satisfies the research objective. The historical archive cannot establish field performance or a mission-wide alarm rate.

## 2. Data contract and provenance

Use the Telemanom release linked by the original authors. Record release location, repository commit, retrieval date, archive and label-file SHA-256 hashes, and one hash per array. Keep the distributed train and test files separate. Use (mission, channel ID) as the logical key. Check duplicate identifiers across missions before mapping a filename to a row. Quarantine ambiguous matches and record the affected denominator. [2, 3]

| Input | Intake check | Action on failure |
|---|---|---|
| Train and test arrays | Both numeric and two-dimensional, equal feature width, finite values, at least one target and sufficient history | Record channel and reason, exclude it from every detector equally |
| Input columns | Column 0 is the telemetry target. Preserve the supplied order of remaining encoded command features. | Stop that channel if the layout cannot be reconciled |
| Expert intervals | Test length equals num_values. Validate zero-based inclusive bounds and one class per interval. Sort interval and class together. | Log the discrepancy. Preserve the source and resolve before scoring |
| Identity and coverage | Match both arrays and metadata. Report missing, duplicate and excluded records separately by mission. | Publish a reconciliation table before interpreting results |

**Known data limitation.** The authors document that released values were scaled using test-set extrema and that timestamps were anonymised. Fitting our transforms on training data cannot undo upstream preprocessing. Our causality claim applies to replay on fixed released arrays. Report latency in readings. Calendar time and physical-unit thresholds cannot be inferred from this release. [2]

## 3. Frozen experiment design

The following are project choices, not published results. Apply the same policy per eligible channel.

1. Partition the official training file in order. For length N use cut points floor(0.70N) and floor(0.85N). Use the first block for fitting, the second for checkpoint selection and the last for calibration. Assign windows by target index. Earlier in-file context may cross a cut point. Never join train and test histories. Require at least one fitting window, one validation target and 100 eligible calibration scores. Record short channels.
2. At index t, forecast telemetry column 0 from rows t−250 through t−1, using all released features. Observe the actual telemetry at t, then compute its score. The first eligible index in a fresh file is 250. Use the identical eligible index range for all three detectors.
3. Compare three per-channel scores. **Level alarm:** absolute deviation from the fitting mean divided by fitting standard deviation, floored at 1e−8. **Persistence:** absolute difference from the previous value. **LSTM:** absolute one-step forecast error. Use released inputs as supplied. The persistence comparison tests whether learning adds value beyond assuming the last reading will repeat.
4. Use two LSTM layers of 80 hidden units, dropout 0.30 between layers and a scalar output. Fit with mean squared error, Adam at 0.001, batch size 64, seed 42, up to 50 epochs. Stop after five epochs without improved validation loss and retain the earliest checkpoint with minimum loss. Record actual epochs, environment and runtime.
5. Apply identical trailing exponential smoothing to all scores with span 30 (weight 2/31 on the current score). Initialise each calibration or test score sequence from its first eligible score. Set each detector’s threshold to its own calibration score at sorted rank ceil(0.99n), counting from one. Freeze the numerical threshold before testing. A strict exceedance raises an alarm. A constant calibration score therefore has a defined threshold. The common percentile is a fair calibration policy; actual test alarm rates may differ.
6. Consecutive flagged indices form one alarm with inclusive endpoints. Each unflagged index closes a run. Log the first flag immediately. No retrospective pruning. Save scores, thresholds, file indices and flags before the evaluator reads test labels. Record a configuration hash before the first test replay.

## 4. Evaluation contract

**Matching.** Work within one mission and channel. Clip truth intervals to the common scored range. Record fully excluded and partly clipped events. For sorted alarms and truths, create an edge for each inclusive overlap. Select a maximum-cardinality one-to-one matching. Among ties minimise total delay from the first observable truth index to the first overlapping flagged index, then choose the lexicographically earliest list of (truth start, alarm start, source row order). Unmatched alarms count as false alarms relative to the supplied labels. Unmatched eligible truths count as misses.

| Measure | Definition |
|---|---|
| Event recall | Matched events / eligible labelled events |
| Event precision | Matched alarms / all alarm episodes |
| False alarms per 1,000 readings | 1,000 × unmatched alarm episodes / all scored readings |
| Fraction of time flagged | Flagged readings / all scored readings |
| Detection delay | Median and 90th percentile of first overlapping flag minus observable event start, among matched events. Report matched and missed counts beside delays. |

Pool counts before computing mission and overall metrics. Report recall by point versus contextual class, with denominators. False alarms have no expert event class. Mark undefined ratios and delays as unavailable. Report results with and without the pilot channel to expose its influence. Use paired, mission-stratified channel resampling (2,000 draws, seed 42) for descriptive 95% percentile intervals on recall and false-alarm differences. Report how many resamples had defined ratios. Channels can share mission incidents, so these intervals describe benchmark variability and do not establish independence across physical faults.

**Scorer example.** With truths [300,305] and [312,316], one alarm [304,314] overlaps both. It can earn one detection only. The other event counts as missed. Scoring must also handle no alarms, no events, boundary-only overlap and two alarms overlapping one event. Verify that extending a replay with future rows leaves all previous scores and flags unchanged.

## 5. Delivery and reproducibility

| Milestone | Evidence required |
|---|---|
| Data intake | Pinned source, file hashes, per-mission channel counts, split boundaries and exclusion ledger |
| Pilot and freeze | Training-side replay on P-1, correct index-250 alignment, prefix-invariance checks, hand-scored event fixtures and frozen configuration |
| Held-out evaluation | Common scored exposure, event ledger and paired comparison for each mission and overall |
| Scientific report | Environment lock, seed, replay instructions and annotated examples of a hit, miss and false alarm where available |

The results table records eligible channels, scored readings, labelled events, alarm episodes, matches, misses, unmatched alarms, precision, recall, false alarms per 1,000 readings, fraction flagged and delay summaries. Link each plot to the underlying channel and indices.

Log protocol changes with their rationale, affected channels and whether test outputs have been inspected. Preserve the original frozen run. Treat variants motivated by observed test performance as exploratory. A new confirmatory claim requires independent evaluation data.

## Sources

1. Project owner. *Space Machine Learning Project Guides*. Supplied Word document, Project 3 and shared ground rules.
2. Hundman and colleagues. [Telemanom project and dataset documentation](https://github.com/khundman/telemanom). Release layout, preprocessing limitations and published benchmark counts.
3. Telemanom authors. [Expert anomaly interval metadata](https://github.com/khundman/telemanom/blob/master/labeled_anomalies.csv). Channel identity, mission, event class and test indices.
4. Hundman, K., Constantinou, V., Laporte, C., Colwell, I. and Soderstrom, T. (2018). [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://doi.org/10.1145/3219819.3219845). KDD 2018. The proposed fixed-threshold comparison is a separate experiment.
