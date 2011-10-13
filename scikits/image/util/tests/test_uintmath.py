import numpy as np

from scikits.image.util.uintmath import is_uint8_like
from scikits.image.util.uintmath import add_uint, subtract_uint
from scikits.image.util.uintmath import multiply_uint, divide_uint


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

