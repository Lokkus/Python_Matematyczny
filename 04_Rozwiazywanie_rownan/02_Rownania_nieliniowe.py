import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def test_funkcja_nielinowa_rozwiazanie():
    x, a, b, c = sp.symbols('x, a, b, c')
    print(sp.solve(a + b*x + c*x**2, x))

def test_szukanie_pierwiastkow():
    # cztery przykladowe f nieliniowe
    x = np.linspace(-2, 2, 1000)
    f1 = x**2 -x -1
    f2 = x**3 - 3 * np.sin(x)
    f3 = np.exp(x) - 2
    f4 = 1 - x**2 + np.sin(50 / (1 + x**2))

    # Wykreslenie kazdej f
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

    for n, f in enumerate([f1, f2, f3, f4]):
        ax[n].plot(x, f, lw=1.5)
        ax[n].axhline(0, ls=':', color='k')
        ax[n].set_ylim(-5, 5)
        ax[n].set_xticks([-2, -1, 0, 1, 2])
        ax[n].set_xlabel(r'$x$', fontsize=18)
    ax[0].set_ylabel(r'$f(x)$', fontsize=18)

    titles=[r'$f(x)=x^2-x-1$', r'$f(x)=x^3-3\sin(x)$', r'$f(x)=\exp(x)-2$', r'$f(x)=\sin\left(50/(1+x^2)\right)+1-x^2$']
    for n, title in enumerate(titles):
        ax[n].set_title(title)
    plt.show()


def test_metoda_bisekcji():
    f = lambda x: np.exp(x) - 2
    tol = 0.1
    a, b = -2, 2
    x = np.linspace(-2.1, 2.1, 1000)

    #wykres f
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(x, f(x), lw=1.5)
    ax.axhline(0, ls=':', color='black')
    ax.set_xticks(np.linspace(-2, 2, 5))
    ax.set_xlabel(r'$x$', fontsize=18)
    ax.set_ylabel(r'$f(x)$', fontsize=18)
    #plt.show()

    # poszukiwanie pierwiastka metoda bisekcji wraz z zaznaczeniem kolejnych krokow na wykresie
    fa, fb = f(a), f(b)
    ax.plot(a, fa, 'ko')
    ax.plot(b, fb, 'ko')
    ax.text(a, fa + 0.5, r'$a$', ha='center', fontsize=18)
    ax.text(b, fb + 0.5, r'$b$', ha='center', fontsize=18)
    n = 1
    while b-a > tol:
        m = a + (b - a)/2
        fm = f(m)
        ax.plot(m, fm, 'ko')
        ax.text(m, fm - 0.5, r'$m_%d$' % n, ha='center')
        n+=1

        if np.sign(fa) == np.sign(fm):
            a, fa = m, fm
        else:
            b, fb = m, fm

    plt.show()






if __name__ == '__main__':
    #test_funkcja_nielinowa_rozwiazanie()
    #test_szukanie_pierwiastkow()
    test_metoda_bisekcji()