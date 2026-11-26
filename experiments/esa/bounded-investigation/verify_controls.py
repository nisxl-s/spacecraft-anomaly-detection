"""Check exact expiry and subsequent reopening."""
import numpy as np

from run import HORIZON, bounded_records

active = np.repeat([False, True, False, True], [1, HORIZON + 11, 2, 3])
starts = np.flatnonzero(bounded_records(active))
np.testing.assert_array_equal(starts, [*range(1, HORIZON + 1),
                                      *range(HORIZON + 14, HORIZON + 17)])
print("Seven-day expiry and reopening after clearance passed.")
