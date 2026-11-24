# ESA Mission 1 frozen temporal extension

The primary comparison is residual versus direct LSTM. Advancement requires the residual model to detect every distinct new incident detected by direct LSTM, equal or exceed its pooled channel-event recall, and have a lower pooled unmatched-alarm rate. The operational budget is assessed separately against two emitted episodes per 1,000 eligible channel-hours. The original 5% flagged-hours limit is also reported.

All four models and their thresholds are fixed to the prior alarm-budget study. The new targets run from 2009-01-01 01:00 UTC through 2013-12-31 23:00 UTC. Replay begins in 2008 to preserve model history, smoothing, and alarm state. The replay verifies the original 2008 raw and emitted flags and saves hashes before opening annotations for matching. The supplied original label catalog and channel files retain their frozen input checksums. Event annotations are available as a catalog of intervals, without a separate per-hour completeness guarantee.

Matching is one-to-one per channel over the entire extension. Channel-events and detection delay are attributed by first observable onset, unmatched alarms by observable alarm start, and operational episodes by streaming alarm start. Calendar-year exposure and flagged time are assigned per hour. An ESA annotation ID denotes one distinct incident across channels. Carry-in events and quality-observable carry-in alarms are omitted from primary new counts and their overlaps are recorded. Episode and event endpoints may be right-censored at the final replay hour.

The execution sources use the original ESA pilot, training checkpoint and fixed-budget workspace inputs. Summary and ledger JSONs, frozen settings, raw replay arrays and audit output can be inspected independently.
