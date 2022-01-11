
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import sympy as sp

def test_wykres_funkcji_i_pochodnych():
    x = np.linspace(-5, 2, 100)
    y1 = x**3 + 5*x**2 + 10
    y2 = 3*x**2 + 10*x
    y3 = 6*x + 10

    fig, ax = plt.subplots()
    ax.plot(x, y1, color='blue', label='y(x)')
    ax.plot(x, y2, color='red', label="y'(x)")
    ax.plot(x, y3, color='green', label="y''(x)")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()

def test_wykres_funkcji_i_pochodnych_z_uzyciem_sympy():
    x = sp.symbols('x')
    expr = x**3 + 5*x**2 + 10
    first_deriv = expr.diff(x)
    second_deriv = first_deriv.diff(x)
    xx = np.linspace(-5, 2, 100)
    yy = sp.lambdify(x, expr)(xx)
    zz = sp.lambdify(x, first_deriv)(xx)
    aa = sp.lambdify(x, second_deriv)(xx)

    fig, ax = plt.subplots()
    ax.plot(xx, yy, color='blue', label='y(x)')
    ax.plot(xx, zz, color='red', label="y'(x)")
    ax.plot(xx, aa, color='green', label="y''(x)")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()


def test_klasa_figure():
    fig = plt.figure(figsize=(8, 2.5), facecolor='#f1f1f1')
    left, bottom, width, heigh = 0.1, 0.1, 0.8, 0.8

    ax = fig.add_axes((left, bottom, width, heigh), facecolor='#e1e1e1')

    x = np.linspace(-2, 2, 1000)
    y1 = np.cos(40*x)
    y2 = np.exp(-x**2)

    ax.plot(x, y1 * y2)
    ax.plot(x, y2, 'g')
    ax.plot(x, -y2, 'g')

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()


def test_wlasciwosci_wykresu():
    x = np.linspace(-5, 5, 5)
    y = np.ones_like(x)

    def axes_settings(fig, ax, title, ymax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, ymax+1)
        ax.set_title(title)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    #grubosc linii
    linewidths = [0.5, 1.0, 2.0, 4.0]

    for n, linewidth in enumerate(linewidths):
        axes[0].plot(x, y+n, color='blue', linewidth=linewidth)
    axes_settings(fig, axes[0], 'Grubosc linii', len(linewidths))

    #styl linii
    linestyles = ['-', '-.', ':']
    for n, linestyle in enumerate(linestyles):
        axes[1].plot(x, y + n, color='blue', lw=2, linestyle=linestyle)

    #wlasny styl linii przerywanej
    line, = axes[1].plot(x, y + 3, color='blue', lw=2)
    length1, gap1, length2, gap2 = 10, 7, 20, 7
    line.set_dashes([length1, gap1, length2, gap2])
    axes_settings(fig, axes[1], 'Rodzaj linii', len(linestyles) + 1)

    #rodzaje markerów
    markers = ['+', 'o', '*', 's', '.', '1', '2', '3', '4']
    for n, marker in enumerate(markers):
        axes[2].plot(x, y+n, color='blue', lw=2, ls='', marker=marker)
    axes_settings(fig, axes[2], 'Markery', len(markers))


    #rozmiar markerow i ich kolor
    markersizecolors = [(4, 'black'), (8, 'red'), (12, 'yellow'), (16, 'lightgreen')]
    for n, (markersize, markerfacecolor) in enumerate(markersizecolors):
        axes[3].plot(x, y+n, color='blue', lw=1, ls='-', marker='o', markersize=markersize, markerfacecolor=markerfacecolor, markeredgewidth=2)
    axes_settings(fig, axes[3], 'Rozmiar i kolor markerow', len(markersizecolors))

    plt.show()


def test_sin_taylor():
    sym_x = sp.Symbol('x')
    x = np.linspace(-2*np.pi, 2*np.pi, 100)

    def sin_expansion(x, n):
        return sp.lambdify(sym_x, sp.sin(sym_x).series(n=n+1).removeO(), 'numpy')(x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, np.sin(x), linewidth=4, color='red', label='sin(x)')

    colors = ['blue', 'black']
    linestyles = [':', '-.', '--']
    for idx, n in enumerate(range(1, 12, 2)):
        ax.plot(x, sin_expansion(x, n), color=colors[idx//3], linestyle=linestyles[idx % 3], linewidth=3, label=f'Przyblizenie rzedu O{n+1}')

    #utworzenie miejsca na legende
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(-1.5*np.pi, 1.5*np.pi)
    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0)
    fig.subplots_adjust(right=.75)
    plt.show()


def test_etykiety_osi_i_tytuly():
    fig, ax = plt.subplots(figsize=(16, 6), subplot_kw={'facecolor': "#ebf5ff"})

    x = np.linspace(0, 50, 500)
    ax.plot(x, np.sin(x) * np.exp(-x/10), lw=1)

    ax.set_xlabel('x', labelpad=5, fontsize=18, fontname='serif', color='blue')
    ax.set_ylabel('f(x)', labelpad=15, fontsize=18, fontname='serif', color='blue')
    ax.set_title('Przykladowe nazwy osi i tytul wykresu', loc='left', fontsize=16, fontname='serif', color='blue')

    plt.show()


def test_zakresy_osi():
    x = np.linspace(0, 30, 500)
    y = np.sin(x) * np.exp(-x/10)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'facecolor': '#ebf5ff'})
    axes[0].plot(x, y, lw=2)
    axes[0].set_xlim(-5, 35)
    axes[0].set_ylim(-1, 1)
    axes[0].set_title('set_xlim/set_ylim')

    axes[1].plot(x, y, lw=2)
    axes[1].axis('tight')
    axes[1].set_title('axis tight')

    axes[2].plot(x, y, lw=2)
    axes[2].axis('equal')
    axes[2].set_title('axis(equal)')
    plt.show()


def test_linie_siatki():
    x = np.linspace(-2 * np.pi, 2*np.pi, 1000)
    y = np.sin(x) * np.exp(-x**2/20)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x_major_ticker = mpl.ticker.MultipleLocator(4)
    x_minor_ticker = mpl.ticker.MultipleLocator(1)
    y_major_ticker = mpl.ticker.MultipleLocator(0.5)
    y_minor_ticker = mpl.ticker.MultipleLocator(0.25)

    for ax in axes:
        ax.plot(x, y, lw=2)
        ax.xaxis.set_major_locator(x_major_ticker)
        ax.yaxis.set_major_locator(y_major_ticker)
        ax.xaxis.set_minor_locator(x_minor_ticker)
        ax.yaxis.set_minor_locator(y_minor_ticker)

    axes[0].set_title('Standardowa siatka')
    axes[0].grid()

    axes[1].set_title('Siatka na podzialkach\n glownych i dodatkowych')
    axes[1].grid(color='blue', which='both', linestyle=':', linewidth=0.5)

    axes[2].set_title('Rozny styl linii x/y dla \npodzialek glownych i standardowych')
    axes[2].grid(color='grey', which='major', axis='x', linestyle='-', linewidth=0.5)
    axes[2].grid(color='grey', which='minor', axis='x', linestyle=':', linewidth=0.25)
    axes[2].grid(color='grey', which='major', axis='y', linestyle='-', linewidth=0.5)
    plt.show()


def test_notacja_naukowa_podzialki():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    x = np.linspace(0, 1e5, 100)
    y = x**2

    axes[0].plot(x, y, 'b.')
    axes[0].set_title('Standardowe etykiety podzialek', loc='right')

    axes[1].plot(x, y, 'b.')
    axes[1].set_title('Etykiety w notacji naukowej', loc='right')

    formatter = mpl.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].yaxis.set_major_formatter(formatter)
    plt.show()


def test_wykresy_w_skali_logarytmicznej():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.linspace(0, 1e3, 100)
    y1, y2 = x**3, x**4

    axes[0].set_title('loglog')
    axes[0].loglog(x, y1, 'b', x, y2, 'r')

    axes[1].set_title('semilog')
    axes[1].semilogy(x, y1, 'b', x, y2, 'r')

    axes[2].set_title('plot / set_xscale / set_yscale')
    axes[2].plot(x, y1, 'b', x, y2, 'r')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    plt.show()


def test_wykresy_z_podwojnymi_osiami():
    fig, axes = plt.subplots(figsize=(8, 4))

    r = np.linspace(0, 5, 100)
    a = 4 * np.pi * r ** 2
    v = (4 * np.pi / 3) * r ** 3

    axes.set_title('Powierzchnia i objetosc kuli', fontsize=16)
    axes.set_xlabel('Promien [m]', fontsize=16)

    axes.plot(r, a, lw=2, color='blue')
    axes.set_ylabel(r'Powierzchnia ($m^2$)', fontsize=16, color='blue')
    for label in axes.get_yticklabels():
        label.set_color('blue')

    axes2 = axes.twinx()
    axes2.plot(r, v, lw=2, color='red')
    axes2.set_ylabel(r'Objetosc ($m^3$)', fontsize=16, color='red')
    for label in axes2.get_yticklabels():
        label.set_color('red')
    plt.show()


if __name__ == '__main__':
    #test_wykres_funkcji_i_pochodnych()
    #test_wykres_funkcji_i_pochodnych_z_uzyciem_sympy()
    #test_klasa_figure()
    #test_wlasciwosci_wykresu()
    #test_sin_taylor()
    #test_etykiety_osi_i_tytuly()
    #test_zakresy_osi()
    #test_linie_siatki()
    #test_notacja_naukowa_podzialki()
    #test_wykresy_w_skali_logarytmicznej()
    test_wykresy_z_podwojnymi_osiami()
