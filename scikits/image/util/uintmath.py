import numpy as np


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
    """Add two uint arrays. Overflow is clipped."""
    assert is_uint8_like(a) and is_uint8_like(b)

    if np.isscalar(b):
        return np.uint8(np.clip(a, MIN_UINT, MAX_UINT - b) + b)
    elif np.isscalar(a):
        return np.uint8(np.clip(b, MIN_UINT, MAX_UINT - a) + a)


def subtract_uint(a, b):
    """Subtract two uint arrays. Underflow is clipped."""
    assert is_uint8_like(a) and is_uint8_like(b)

    if np.isscalar(b):
        return np.uint8(np.clip(a, MIN_UINT + b, MAX_UINT) - b)
    elif np.isscalar(a):
        return np.uint8(a - np.clip(b, MIN_UINT,  MIN_UINT + a))


def multiply_uint(a, b):
    """Multiply two uint arrays. Overflow is clipped."""
    assert is_uint8_like(a) and is_uint8_like(b)

    if np.isscalar(a) and np.isscalar(b):
        maxval = np.floor(MAX_UINT / b)
        if a > maxval:
            return np.uint8(MAX_UINT)
        return npint8(a * b)

    if np.isscalar(a):
        a, b = b, a

    if np.isscalar(b):
        maxval = np.floor(MAX_UINT / b)
        mask = a <= maxval
        c = np.empty(a.shape, dtype=np.uint8)
        c[~mask] = MAX_UINT
        c[mask] = a[mask] * b
    return c


def divide_uint(a, b):
    """Divide two uint arrays. Output is rounded (as opposed to floored)."""
    assert is_uint8_like(a) and is_uint8_like(b)
    return np.uint8((a + b / 2) / b)


def test_is_uint8_like():
    assert is_uint8_like(np.arange(0, 256, 25))
    assert is_uint8_like(255)
    assert not is_uint8_like(256)
    assert not is_uint8_like(-1)
    assert not is_uint8_like(1.)

def test_add_scalars():
    a = 150
    assert add_uint(a, a) == 255

def test_subtract_scalars():
    assert subtract_uint(100, 101) == 0

def test_multiply_scalars():
    assert multiply_uint(100, 3) == 255

def test_divide_scalars():
    assert divide_uint(7, 5) == 1
    assert divide_uint(8, 5) == 2


def test_add_array_scalar():
    a = np.arange(100, 251, 50)
    b = 50
    c = (150, 200, 250, 255)
    assert all(add_uint(a, b) == c)
    assert all(add_uint(b, a) == c)

def test_subtract_array_scalar():
    a = np.arange(0, 101, 25)
    b = 50
    c = (0, 0, 0, 25, 50)
    d = (50, 25, 0, 0, 0)
    assert all(subtract_uint(a, b) == c)
    assert all(subtract_uint(b, a) == d)

def test_multiply_array_scalar():
    a = np.arange(0, 5, dtype=np.uint8)
    b = 100
    c = (0, 100, 200, 255, 255)
    assert all(multiply_uint(a, b) == c)
    assert all(multiply_uint(b, a) == c)

def test_divide_array_scalar():
    a = np.arange(0, 10, dtype=np.uint8)
    b = 3
    c = (0, 0, 1, 1, 1, 2, 2, 2, 3, 3)
    assert all(divide_uint(a, b) == c)


if __name__ == '__main__':
    import nose
    nose.runmodule()

