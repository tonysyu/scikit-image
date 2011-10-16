import numpy as np


__all__ = ['add_uint', 'subtract_uint', 'multiply_uint', 'divide_uint']

MIN_UINT = 0
MAX_UINT = 255


def is_uint8_like(a):
    """Return True if input is an integer and casts to uint8 w/o truncation."""
    if np.any(a < MIN_UINT) or np.any(a > MAX_UINT):
        return False
    if hasattr(a, 'dtype') and np.issubdtype(a.dtype, np.integer):
        return True
    if isinstance(a, int):
        return True
    else:
        return False


def add_uint(a, b):
    """Add two uint8 arrays. Overflow is clipped."""
    assert is_uint8_like(a) and is_uint8_like(b)

    if np.isscalar(b):
        return np.uint8(np.clip(a, MIN_UINT, MAX_UINT - b) + b)
    elif np.isscalar(a):
        return np.uint8(np.clip(b, MIN_UINT, MAX_UINT - a) + a)

    c = np.empty(a.shape, dtype=np.uint8)
    mask = a < (MAX_UINT - b)
    c[~mask] = MAX_UINT
    c[mask] = a[mask] + b[mask]
    return c


def subtract_uint(a, b):
    """Subtract two uint8 arrays. Underflow is clipped."""
    assert is_uint8_like(a) and is_uint8_like(b)

    if np.isscalar(b):
        return np.uint8(np.clip(a, MIN_UINT + b, MAX_UINT) - b)
    elif np.isscalar(a):
        return np.uint8(a - np.clip(b, MIN_UINT,  MIN_UINT + a))

    c = np.empty(a.shape, dtype=np.uint8)
    mask = a > (MIN_UINT + b)
    c[~mask] = MIN_UINT
    c[mask] = a[mask] - b[mask]
    return c


def multiply_uint(a, b):
    """Multiply two uint8 arrays. Overflow is clipped."""
    assert is_uint8_like(a) and is_uint8_like(b)

    if np.isscalar(a) and np.isscalar(b):
        maxval = np.floor(MAX_UINT / b)
        if a > maxval:
            return np.uint8(MAX_UINT)
        return npint8(a * b)

    if np.isscalar(a):
        a, b = b, a

    c = np.empty(a.shape, dtype=np.uint8)
    mask = a <= MAX_UINT / b
    c[~mask] = MAX_UINT
    if not np.isscalar(b):
        b = b[mask]
    c[mask] = a[mask] * b
    return c


def divide_uint(a, b):
    """Divide two uint8 arrays. Output is rounded (as opposed to floored)."""
    assert is_uint8_like(a) and is_uint8_like(b)
    return np.uint8((a + b / 2) / b)

