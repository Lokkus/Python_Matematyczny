import sympy as sp
x, y, z = sp.symbols('x, y, z')

def test_pochodne():
    f = sp.Function('f')(x)
    print(sp.diff(f, x))


def test_pochodne_wielomiany():
    expr = x**4 +x**3 + x**2 + 1
    print(expr.diff(x))
    print(expr.diff(x, x))


def test_pochodne_f_trygonometrycznych():
    expr = sp.sin(x*y)*sp.cos(x/2)
    print(expr.diff(x))


def test_calki():
    a, b, x, y = sp.symbols('a, b, x, y')
    f = sp.Function('f')(x)
    print(sp.integrate(f))

    print(sp.integrate(sp.sin(x)))

    expr = sp.sin(x*sp.exp(y))
    print(sp.integrate(expr, x))


def test_granice():
    print(sp.limit(sp.sin(x)/x, x, 0))

    f = sp.Function('f')
    y, h = sp.symbols('y, h')
    diff_limit = (f(y+h) - f(y))/h
    print(f'Pochodna cos(y): {sp.limit(diff_limit.subs(f, sp.cos), h, 0)}')
    print(f'Pochodna sin(y): {sp.limit(diff_limit.subs(f, sp.sin), h, 0)}')

    #licznie asymptoty ukosnej
    expr = (x**2 - 3*x) / (2*x - 2)
    #y = ax + b
    a = sp.limit(expr/x, x, sp.oo)
    b = sp.limit(expr - a*x, x, sp.oo)
    print(f'a = {a}, b = {b}')


def test_sumy_i_iloczyny_uogolnione():
    n = sp.symbols('n', integer=True)
    x = sp.Sum(1/(n**2), (n, 1, sp.oo))
    print(x.doit())
    x = sp.Product(n, (n, 1, 7))
    print(x.doit())



if __name__ == '__main__':
    #test_pochodne()
    #test_pochodne_wielomiany()
    #test_pochodne_f_trygonometrycznych()
    #test_calki()
    #test_granice()
    test_sumy_i_iloczyny_uogolnione()