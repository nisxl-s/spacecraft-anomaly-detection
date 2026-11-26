# NASA sustained records and paging development replay

Use the previously examined 78-channel NASA SMAP/MSL labeled replay scores. Keep residual LSTM weights, frozen thresholds, two-crossing confirmation, original cap and rearm identical for the baseline. The sustained stream records a confirmed run until the score clears. The paging trial keeps all original pages and adds one reminder at the cap boundary (30 readings after the sustained run starts) if the score is still above threshold. Duplicate timestamps are counted once per channel.

Event scoring uses the existing one-to-one matcher. Coverage uses alarm or record intervals. Timely page detection requires the page timestamp itself to fall within the labeled interval. NASA channels are scored independently on sample indices.

Use the NASA benchmark modules on the Python import path and run `python run.py /path/to/heldout /path/to/output`. The script verifies score hashes, original flags, and original event matches against the held-out replay. The summary and ledgers contain the development results.
