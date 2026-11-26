# Sustained anomaly record validation

The paired ESA Mission 2 trials compare the original paging policy and uncapped investigation records on the same hourly persistence scores. The original NASA LSTM is outside this transfer test. The NASA historical replay provides a separate regression check.

## Reproduce

Use NumPy 2 and pandas 2.2. Put the verified official Mission 2 v2 channel ZIPs and the labels.csv and anomaly_types.csv annotation files under esa-mission2/inputs next to this directory. The dataset release is https://zenodo.org/records/15237121. The input ZIPs contain official pandas tables. Use the trusted release members and compare their digests against the two seal.json manifests before running the scripts.

```sh
python nasa-independent-validation/verify_controls.py
python nasa-independent-validation/run.py freeze
python nasa-independent-validation/run.py score
python nasa-independent-validation/followup/run.py freeze
python nasa-independent-validation/followup/run.py score
```

Each seal authenticates the protocol, scoring code, paging rule, channel inputs and alarm arrays. The freeze command writes the alarm arrays without opening labels. The scoring command verifies each frozen array before matching it to the annotations. The event ledgers show official IDs, interval boundaries and eligibility decisions. The summary counts overlap with each labeled channel-event independently. The same record can overlap multiple labels, so these counts use a different matching convention from the original NASA replay.

The independent event coverage denominator counts only eligible events with a scored hourly sample inside the labeled interval. An in-event page requires a page timestamp inside that interval. The pooled occupancy denominator is eligible channel-hours. The 10% occupancy threshold is provisional.

## Results

- Original NASA archive replay reproduced the complete 78-channel action ledger, including all 2,193 original page indices. The deterministic cap and five-low rearm control also passed.
- Initial ESA panel (channels 15–20) contained three post-split anomaly channel-events associated with a previously studied incident. After excluding those rows, independent event coverage was undefined. Both policies occupied 386 of 91,866 hours, and the longest record lasted 13 hours.
- Follow-up panel (channels 81, 97, 96) contained five eligible channel-events from five distinct incidents. Capped flags overlapped two and sustained records overlapped three. The extra overlap was id_615 on channel_97. Its existing record had already been active for 89 hours when the first in-event sample arrived.
- The 142 original page timestamps were preserved. Pooled occupancy increased from 0.76% to 11.30%, above the 10% gate. The longest record lasted 2,822 hours (about 118 days). The sample was deliberately enriched for anomalies, so these results do not estimate mission-wide coverage or occupancy.

Keep the paging policy and treat sustained records as investigation evidence. Raw uncapped records did not pass the provisional burden gate.
