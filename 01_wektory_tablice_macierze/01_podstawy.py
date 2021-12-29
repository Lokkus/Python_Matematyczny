import numpy as np


def test_podstawy():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    print(type(data))
    print(data)
    print(data.ndim)
    print(data.shape)
    print(data.dtype)
    print(data.nbytes)


def test_typy():
    data = np.array([[1, 2], [6, 7]], dtype=int)
    print(data)
    print(data.dtype)

    data = np.array([[1, 2], [6, 7]], dtype=float)
    print(data)
    print(data.dtype)

    data = np.array([[1, 2], [6, 7]], dtype=complex)
    print(data)
    print(data.dtype)


def test_sqrt_complex():
    print(np.sqrt(np.array([-4, 0, 1], dtype=complex)))


def test_zeros_ones():
    print(np.zeros((2, 3)))
    print(np.ones(4))
    print(5 * np.ones((4, 4), dtype=int))

    print(np.full((5, 5), 7.2))


def test_wypelnianie_tablic():
    print(np.arange(0, 10, 1, dtype=int))
    print(np.linspace(0.0, 0.1, 11))

    print(np.logspace(0, 2, 100))

    x = np.array([-1, 0, 1])
    y = np.array([-2, 0, 2])
    X, Y = np.meshgrid(x, y)
    print((X + Y)**2)


def test_macierze():
    print(np.identity(4, dtype=int))
    print(np.eye(3, k=1))


if __name__ == '__main__':
    #test_podstawy()
    #test_typy()
    #test_sqrt_complex()
    #test_zeros_ones()
    #test_wypelnianie_tablic()
    test_macierze()