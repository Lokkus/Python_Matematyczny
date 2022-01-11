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


def test_wycinki():
    f = lambda m, n: n+10 * m
    A = np.fromfunction(f, (6, 6), dtype=int)
    print(A)

    print(f'Druga kolumna:\n {A[:,1]}')
    print(f'Drugi wiersz:\n {A[1:,]}')
    print(f'Usuniecie pierwszego wiersza i pierwszej kolumny:\n {A[1:, 1:]}')
    print(f'Usuniecie ostatniego wiersza i pierwszej kolumny:\n {A[:-1, 1:]}')
    print(f'Usuniecie drugiego wiersza i drugiej kolumny:\n {np.delete(np.delete(A, 1, axis=0), 1, axis=1)}')
    print(f'Co drugi wiersz poczawszy od elementy [0,0]:\n {A[::2, ::2]}')

def test_widoki():
    f = lambda x, y: x+10 * y
    A = np.fromfunction(f, (5, 5), dtype=int)
    print(f'Oryginal:\n{A}')

    B = A[1:4, 1:4]
    print(f'Widok:\n {B}')
    B[:, :] = 0
    print(f'Oryginal po zmianie:\n {A}')

    print('To samo tyle ze z uzyciem copy')

    C = np.fromfunction(f, (6, 6), dtype=int)
    D = C[1:3, 1:3].copy()
    print(f'Oryginal:\n {C}\nPodmacierz: \n{D}')
    D[:,:] = 1
    print(f'Oryginal po zmianach:\n {C}\nPodmacierz po zmianach: \n{D}')


def test_fancy_indexing():
    f = lambda x, y: x + 10 * y
    A = np.fromfunction(f, (5, 5), dtype=int)
    print(f'Oryginal:\n{A}')
    print(f'Pokaz 2 i 4 kolumne:\n {A[:, [2,4]]}')
    print(f'Pokaz 0 i 4 wiersz oraz 1 i 3 kolumne:\n {A[[0, 4], [1, 3]]}')

    A = 5*np.arange(10)
    indices = [2, 4, 6]
    B = A[indices]
    print(f'Oryginal A:\n{A}')
    print(f'B:\n{B}')
    A[indices] = -10
    print(f'Oryginal A po zmianach:\n{A}')
    B = A[A > 10]
    print(f'B po zmianach\n{B}')


def test_zmiana_ksztaltu_i_rozmiaru():
    data = np.array([[1, 2], [5, 6]])
    print(np.reshape(data, (1, 4)))
    print(data.reshape(4))
    print(data.flatten())

    x = np.arange(0, 5)
    col = x[:, np.newaxis]
    row = x[np.newaxis, :]
    print(f'Orginal x: \n{x}, \ncol:\n{col}, \nrow:\n{row}')


def test_laczenie_wektorow_w_tablice():
    print(np.vstack((np.arange(5), np.arange(5, 10), np.arange(10, 15)))) # laczenie wektorow wierszami

    # laczenie kolumnami
    data1 = np.expand_dims(np.arange(5), axis=1)
    data2 = np.expand_dims(np.arange(5, 10), axis=1)
    data3 = np.expand_dims(np.arange(10, 15), axis=1)
    x = np.hstack((data1, data2, data3))
    print(x)


def test_operacje_arytmetyczne():
    x = np.array([[1, 2], [5, 6]])
    y = np.array([[8,7], [4, 2]])

    print(f'Mnozenie przez skalar: Oryginal: \n{x}\nSkalar: {5}\nPo przemnozeniu: \n{5 * x}')
    print(f'Dodawanie tablic\nOrygnal x: \n{x}\nOryginal y:\n{y}\nWynik:\n{x + y}')
    print(f'Mnozenie macierzowe:\n{np.matmul(x, y)}')


def test_funkcje_matematyczne():
    arg = np.arange(0, 2*np.pi, 0.001)
    sin = np.sin(arg)
    cos = np.cos(arg)
    add = np.add(sin**2, cos**2)


def test_vectorize():
    def heaviside(x):
        return 1 if x > 0 else 0

    print(f'heaviside(-1): {heaviside(-1)}')
    print(f'heaviside(-0): {heaviside(0)}')
    print(f'heaviside(1.5): {heaviside(1.5)}')
    # ale
    heaviside = np.vectorize(heaviside)
    x = np.linspace(-5, 5, 11)
    print(heaviside(x))


def test_funkcje_agregujace():
    data = np.random.normal(size=(5, 5))
    print(data)
    print(data.mean())

    data = np.random.normal(size=(3,4,5))
    print(data)
    print(data.sum(axis=0).shape)


def test_funkcje_agregujace_2():
    data = np.arange(1, 10).reshape(3, 3)
    print(data)
    print(f'Suma: {data.sum()}')
    print(f'Suma axis 0: {data.sum(axis=0)}')
    print(f'Suma axis 1: {data.sum(axis=1)}')


def test_wyrazenia_warunkowe():
    a = np.array([1, 2, 3, 4])
    b = np.array([4, 3, 2, 1])
    print(a < b)
    print(np.all(a < b))
    print(np.any(a < b))

    x = np.array([-2, -1, 0, 1, 2])
    print(x > 0)
    print(1 * (x > 0))
    print(1 + (x > 0))


def test_wyrazenia_logiczne_i_warunkowe():
    data = np.linspace(-4, 4, 9)
    print(data)
    print(np.where(data < 0, data**2, data**3))
    print(np.choose([0, 1, 2, 2, 1, 0, 0, 2, 1], [data*2, data*3, data*4]))
    print(np.select([data < -1, data < 2, data >= 2], [data**2, data**3, data**4]))
    print(np.nonzero(abs(data) > 2)) # zwraca indeksy gdzie abs(x) > 2
    print(data[np.nonzero(abs(data) > 2)]) # tutaj sa zwracane wartosci
    print(data[abs(data) > 2]) # to samo co wyzej

def test_operacje_na_zbiorach():
    a = np.unique([1, 2, 3, 3])
    b = np.unique([2, 3, 4, 4, 5, 6, 5])
    print(np.in1d(a, b))
    print(np.intersect1d(a, b))
    print(np.setdiff1d(a, b))
    print(np.union1d(a, b))


def test_operacje_na_tablicach():
    data = np.arange(9).reshape(3, 3)
    print(data)
    print(np.transpose(data))


def test_generate_n_unique_integers():
    def generator(start, stop, n):
        if n >= (start + stop)-1:
            print('Popraw dane, start + stop < n')
            return
        s = set()
        while len(s) < n:
            s.add(np.random.randint(start, stop))
        return s

    print(list(generator(1, 100, 19)))


def test_operacje_macierzowe_i_wektorowe():
    a = np.arange(1, 7).reshape(2, 3)
    print(a)
    b = np.arange(1, 7).reshape(3, 2)
    print(b)
    print(np.dot(a, b))


if __name__ == '__main__':
    #test_podstawy()
    #test_typy()
    #test_sqrt_complex()
    #test_zeros_ones()
    #test_wypelnianie_tablic()
    #test_macierze()
    #test_wycinki()
    #test_widoki()
    #test_fancy_indexing()
    #test_zmiana_ksztaltu_i_rozmiaru()
    #test_laczenie_wektorow_w_tablice()
    #test_operacje_arytmetyczne()
    #test_funkcje_matematyczne()
    #test_vectorize()
    #test_funkcje_agregujace()
    #test_funkcje_agregujace_2()
    #test_wyrazenia_warunkowe()
    #test_wyrazenia_logiczne_i_warunkowe()
    #test_operacje_na_zbiorach()
    #test_operacje_na_tablicach()
    #test_generate_n_unique_integers()
    test_operacje_macierzowe_i_wektorowe()