import sympy as sp

def test_symbol():
    y = sp.Symbol("y")
    print(sp.sqrt(y**2))

def test_duze_liczby():
    i = sp.Integer(19)
    print(i)
    print(i**50)

    #liczby zmiennoprzecinkowe
    x = sp.Float(0.3, 25)
    y = sp.Float('0.3', 25)
    print(x)
    print(y)


def test_liczby_wymierne():
    print(sp.Rational(11, 13))
    x = sp.Rational(2, 3)
    y = sp.Rational(4, 5)
    print(x+y)
    print(x*y)


def test_lambda():
    x = sp.symbols('x')
    h = sp.Lambda(x, x**2)
    print(h(5))
    print(h)


def test_wyrazenia():
    x, y = sp.symbols('x, y')
    expr = 1 + 2*x**2 + 3*x**3
    print(expr.subs(x, 2))

    expr = 2 * (x**2 - x) - x * (x+1)
    print(expr)
    print(sp.simplify(expr))

    # rozwijanie wyrazen
    expr = (x + 1) * (x + 2)
    print(sp.expand(expr))

    # rozwijanie funkcji trygonometrycznych
    expr = sp.sin(x + y).expand(trig=True)
    print(expr)

    # rozwijanie logarytmow
    x, y = sp.symbols('x, y', positive=True)
    print(sp.log(x * y).expand(log=True))


def test_factor_collect_combine():
    x, y, z = sp.symbols('x, y, z')
    print(sp.factor(x**2 - 1))
    print(sp.factor(x * sp.cos(y) + sp.sin(z) * x))

    # collect
    expr = x + y + x * y * z
    print(f'Collect:\n{expr}')
    print(expr.collect(x))
    print(expr.collect(y))


def test_przeksztalcenia_ulamkow():
    x, y, z = sp.symbols('x, y, z')
    print(sp.apart(1/(x**2 + 3*x + 2), x))

    print(sp.together(1 / (y * x + y) + 1 /(1 + x)))

    print(sp.cancel(y / (y * x + y)))


def test_podstawienia():
    x, y, z = sp.symbols('x, y, z')
    print((x + y).subs(x, y))

    expr = x * y + z**2 * x
    val = {x: 1.25, y: 0.4, z: 3.2}
    print(expr.subs(val))


def test_ewaluacja_wyrazen():
    x, y, z = sp.symbols('x, y, z')
    print(sp.N(1 + sp.pi))
    print(sp.N(sp.pi, 50))
    print((x + 1/sp.pi).evalf(10))

    expr = sp.sin(sp.pi * x * sp.exp(x))
    print([expr.subs(x, xx).evalf(3) for xx in range(0, 10)])

    #lambdify
    expr_lam = sp.lambdify(x, expr)
    print(expr_lam(1))


if __name__ == '__main__':
    sp.init_printing()
    #test_duze_liczby()
    #test_liczby_wymierne()
    #test_lambda()
    #test_wyrazenia()
    #test_factor_collect_combine()
    #test_przeksztalcenia_ulamkow()
    #test_podstawienia()
    test_ewaluacja_wyrazen()
