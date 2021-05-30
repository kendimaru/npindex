import pytest
import numpy as np
from npindex import npidx


@pytest.fixture
def a2d():
    side_size = 5

    a = []
    for h in range(side_size):
        for w in range(side_size):
            a.append(f'{h}_{w}')

    nda = np.array(a)
    return nda.reshape(side_size, side_size)


@pytest.fixture
def a3d():
    side_size = 5

    a = []
    for h in range(side_size):
        for w in range(side_size):
            for d in range(side_size):
                a.append(f'{h}_{w}_{d}')

    nda = np.array(a)
    return nda.reshape(side_size, side_size, side_size)


def test_one_param():
    a = np.arange(4)
    assert a[npidx[1]] == a[1]
    assert np.array_equal(a[npidx[1:2]], a[1:2])
    assert np.array_equal(a[npidx[1:3]], a[1:3])


def test_two_params(a2d):
    a = a2d
    assert np.array_equal(a[npidx[2, 2]], a[2, 2])
    assert np.array_equal(a[npidx[2:3, 2:3]], a[2:3, 2:3])
    assert np.array_equal(a[npidx[:, :]], a)


def test_three_params(a3d):
    a = a3d
    assert np.array_equal(a[npidx[1, 2, 3]], a[1, 2, 3])
    assert np.array_equal(a[npidx[:, :, :]], a)



