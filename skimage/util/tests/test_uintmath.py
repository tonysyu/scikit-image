import numpy as np

from skimage.util.uintmath import is_uint8_like
from skimage.util.uintmath import add_uint, subtract_uint
from skimage.util.uintmath import multiply_uint, divide_uint


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
    a = np.arange(0, 5)
    b = 100
    c = (0, 100, 200, 255, 255)
    assert all(multiply_uint(a, b) == c)
    assert all(multiply_uint(b, a) == c)

def test_divide_array_scalar():
    a = np.arange(0, 10)
    b = 3
    c = (0, 0, 1, 1, 1, 2, 2, 2, 3, 3)
    assert all(divide_uint(a, b) == c)


def test_add_arrays():
    a = np.array([(125, 126, 127), (128, 129, 130)])
    c = [[250, 252, 254], [255, 255, 255]]
    assert np.all(add_uint(a, a) == c)

def test_subtract_arrays():
    a = np.array([(1, 2, 3), (4, 5, 6)])
    b = np.array([(6, 5, 4), (3, 2, 1)])
    c = [[0, 0, 0], [1, 3, 5]]
    assert np.all(subtract_uint(a, b) == c)

def test_multiply_arrays():
    a = np.array([(14, 15), (16, 17)])
    c = [(14**2, 15**2), (255, 255)]
    assert np.all(multiply_uint(a, a) == c)

def test_divide_arrays():
    a = np.array([(7, 8), (149, 150)])
    b = np.array([(5, 5), (100, 100)])
    c = [(1, 2), (1, 2)]
    assert np.all(divide_uint(a, b) == c)


def test_add_edge_cases():
    a = np.arange(4)
    c = (253, 254, 255, 255)
    assert all(add_uint(a, 253) == c)

def test_subtract_edge_cases():
    a = np.arange(4)
    c = (2, 1, 0, 0)
    assert all(subtract_uint(2, a) == c)

def test_multiply_edge_cases():
    a = np.array([2, 5, 2])
    b = np.array([127, 51, 128])
    c = (254, 255, 255)
    assert all(multiply_uint(a, b) == c)


if __name__ == '__main__':
    np.testing.run_module_suite()

