"""Verify the 30-reading cap and five-low rearm against known sequences."""
import numpy as np

from frozen_alarm import confirmed_episodes
from run import pages

# A 40-high run crosses the cap. Four lows leave paging locked out,
# then five lows clear the lockout for the last burst of highs.
raw = np.repeat([True, False, True, False, True], [40, 4, 3, 5, 3])
capped = confirmed_episodes(raw)
record = raw & np.r_[False, raw[:-1]]
np.testing.assert_array_equal(np.flatnonzero(capped), [*range(1, 31), 53, 54])
np.testing.assert_array_equal(pages(capped), [1, 53])
np.testing.assert_array_equal(np.flatnonzero(record), [*range(1, 40), 45, 46, 53, 54])
print("Control passed: cap, five-low rearm, sustained tail and unchanged page indices.")
