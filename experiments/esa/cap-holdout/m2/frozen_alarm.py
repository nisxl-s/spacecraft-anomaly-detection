"""The existing NASA paging rule with its original constants."""
import numpy as np

MAX_FLAG_READINGS = 30

CLEAR_READINGS_TO_REARM = 5

CONFIRM_READINGS = 2

def confirmed_episodes(raw_flags, required=CONFIRM_READINGS):
    """Keep output causal. The threshold-crossing reading is never restored."""
    if required < 1:
        raise ValueError("Confirmation requires at least one high reading")
    emitted = np.zeros_like(raw_flags)
    consecutive_high = consecutive_clear = active_readings = 0
    lockout = False
    for i, high in enumerate(raw_flags):
        if lockout:
            consecutive_clear = 0 if high else consecutive_clear + 1
            if consecutive_clear == CLEAR_READINGS_TO_REARM:
                lockout = False
            continue
        if not high:
            consecutive_high = active_readings = 0
            continue
        consecutive_high += 1
        if consecutive_high < required:
            continue
        emitted[i] = True
        active_readings += 1
        if active_readings == MAX_FLAG_READINGS:
            lockout = True
            consecutive_high = consecutive_clear = active_readings = 0
    return emitted
