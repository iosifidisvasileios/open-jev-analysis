"""Probability validation shared by experiment scoring and comparisons."""

import math


def probability(value):
    """Return a finite numeric probability in [0, 1], rejecting invalid input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Expected a numeric probability')
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Probability must be finite and between 0 and 1')
    return float(value)
