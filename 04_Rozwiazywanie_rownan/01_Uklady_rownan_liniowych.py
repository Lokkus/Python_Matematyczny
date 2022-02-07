import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

def test_solve():
    # rozwiazac uklad rownan 2x + 3y = 4; 5x + 4y = 3;
    A = sp.Matrix([[2, 3], [5, 4]])
    b = sp.Matrix([4, 3])
    L, U, _ = A.LUdecomposition()
    print(f'L:{L}')
    print(f'U: {U}')
    print(f'L*U: {L*U}')
    x = A.solve(b)
    print(f'Solve: {x}')


def test_bledy_numeryczne_porownanie():
    # eozwiazujemy i porownujemy rownanie:
    #[[1, sqrt(p)], [1, 1/(sqrt(p))]], [x1, x2], [1, 2]
    p = sp.symbols('p', positive=True)
    A = sp.Matrix([[1, sp.sqrt(p)], [1, 1/sp.sqrt(p)]])
    b = sp.Matrix([1, 2])
    x = A.solve(b)
    print(f'Rozwiazanie: {x}')
    x_sym_sol = A.solve(b)
    print(x_sym_sol)
    print(x_sym_sol.simplify())
    Acond = A.condition_number().simplify()

    # Funkcje do rozwiazania numerycznego
    AA = lambda p: np.array([[1, np.sqrt(p)], [1, 1/np.sqrt(p)]])
    bb = np.array([1, 2])
    x_num_sol = lambda p: np.linalg.solve(AA(p), bb)

    # wykreslenie roznic pomiedzy symbolicznym (dokladnym) rozwiazaniem a rozwiazaniami numerycznymi
    p_vec = np.linspace(0.9, 1.1, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for n in range(2):
        x_sym = np.array([x_sym_sol[n].subs(p, pp).evalf() for pp in p_vec])
        x_num = np.array([x_num_sol(pp)[n] for pp in p_vec])
        axes[0].plot(p_vec, (x_num - x_sym)/x_sym, 'k')
    axes[0].set_title('Blad rozwiazania\n(numeryczne - symboliczne/symboliczne')
    axes[0].set_xlabel(r'$p$', fontsize=18)
    axes[1].plot(p_vec, [Acond.subs(p, pp).evalf() for pp in p_vec])
    axes[1].set_title('Wspoldzynnik uwarunkowania')
    axes[1].set_xlabel('$p$', fontsize=18)
    plt.show()




if __name__ == '__main__':
    #test_solve()
    test_bledy_numeryczne_porownanie()