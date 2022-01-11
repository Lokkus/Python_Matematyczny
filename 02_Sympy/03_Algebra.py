import sympy as sp


def test_rozwiazanie_rownania():
    x = sp.Symbol('x')
    expr = x**2 + 2*x -3
    print(sp.solve(expr))

    expr = sp.sin(x) - sp.cos(x)
    print(sp.solve(expr))


def test_uklad_rownan():
    x, y = sp.symbols('x, y')
    eq1 = x + 2 * y - 1
    eq2 = x - y + 1
    print(sp.solve([eq1, eq2], [x, y], dict=True))

    eq1 = x**2 - y
    eq2 = y**2 -x
    print(sp.solve([eq1, eq2], [x, y], dict=True))


def test_algebra_liniowa():
    print(sp.Matrix([[1, 2], [4, 3]]))
    print(sp.Matrix(3, 4, lambda m, n: 10 * m + n))


if __name__ == '__main__':
    #test_rozwiazanie_rownania()
    #test_uklad_rownan()
    test_algebra_liniowa()