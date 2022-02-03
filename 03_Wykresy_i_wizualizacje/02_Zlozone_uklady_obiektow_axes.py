import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import sympy as sp

def test_wstawki():
    fig = plt.figure(figsize=(8, 4))
    def f(x):
        return 1/(1+x**2) + 0.1/(1 + ((3 - x)/0.1)**2)

    def plot_and_format_axes(ax, x, f, fontsize):
        ax.plot(x, f(x), linewidth=2)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(15)) # gestosc podzialki
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(14)) # gestosc podzialki
        ax.set_xlabel(r'$x$', fontsize=fontsize)
        ax.set_ylabel(r'$f(x)$', fontsize=fontsize)

    #glowny wykres
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.8], facecolor='#ffffff')
    x = np.linspace(-4, 14, 1000)
    plot_and_format_axes(ax, x, f, 18)


    #Wstawka
    x0, x1 = 2.5, 3.5
    ax.axvline(x0, ymax=0.3, color='grey', linestyle=':')
    ax.axvline(x1, ymax=0.3, color='grey', linestyle=':')

    ax = fig.add_axes([0.5, 0.5, 0.38, 0.42], facecolor='none')
    x = np.linspace(x0, x1, 1000)
    plot_and_format_axes(ax, x, f, 14)

    plt.show()


def test_wstawki_subplots():
    fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True, squeeze=False)

    x1 = np.random.randn(100)
    x2 = np.random.randn(100)

    axes[0, 0].set_title('Wartosci nieskorelowane')
    axes[0, 0].scatter(x1, x2)

    axes[0, 1].set_title('Slaba korelacja dodatnia')
    axes[0, 1].scatter(x1, x1 + x2)

    axes[1, 0].set_title('Slaba korelacja ujemna')
    axes[1, 0].scatter(x1, -x1 + x2)

    axes[1, 1].set_title('Wartosci silnie skorelowane')
    axes[1, 1].scatter(x1, x1 + 0.15 * x2)

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.1, hspace=0.2)
    plt.show()


def test_wykresy_colormap():
    x = y = np.linspace(-10, 10, 150)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(X) * np.cos(Y) * np.exp(-(X/5)**2 - (Y/5)**2)

    fig, ax = plt.subplots(figsize=(6, 5))
    p = ax.pcolor(X, Y, Z, vmin=-abs(Z).max(), vmax=abs(Z).max(), cmap=mpl.cm.bwr)

    ax.axis('tight')
    ax.set_xlabel(r'$x$', fontsize=18)
    ax.set_ylabel(r'$y$', fontsize=18)

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

    cb = fig.colorbar(p, ax=ax)
    cb.set_label(r'$z$', fontsize=18)
    cb.set_ticks([-1, -.5, 0, .5, 1])
    plt.show()

if __name__ == '__main__':
    #test_wstawki()
    #test_wstawki_subplots()
    test_wykresy_colormap()