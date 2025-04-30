import numpy as np
from prettytable import PrettyTable

def tikslo_funkcija(x):
    return -x[0] * x[1] * x[2]

def gi(x):
    return 2 * (x[0]*x[1] + x[1]*x[2] + x[0]*x[2]) - 1

def hi(x):
    return [-x[0], -x[1], -x[2]]

def baudos_funkcija(x, r):
    g = gi(x)
    h = hi(x)
    bauda = g**2 + sum((max(0, h_i))**2 for h_i in h)
    return tikslo_funkcija(x) + 1.0/r * bauda

def baudos_funkcijos_gradientas(x, r):
    grad_f = np.array([
        -x[1]*x[2],
        -x[0]*x[2],
        -x[0]*x[1]
    ])
    g = gi(x)
    grad_g = np.array([
        4 * g * (x[1] + x[2]),
        4 * g * (x[0] + x[2]),
        4 * g * (x[0] + x[1])
    ])
    h = hi(x)
    grad_h = np.zeros(3)
    for i in range(3):
        if h[i] > 0:
            grad_h[i] = -2 * h[i]
    return grad_f + (1.0 / r) * (grad_g + grad_h)

def tenkina_apribojimus(x):
    return gi(x) <= 0 and all(xi >= 0 for xi in x)

def auksinio_pjuvio_paieska(f, x, grad, r, a=0, b=10, tol=1e-9):
    phi = (np.sqrt(5) - 1) / 2
    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)
    fx1 = f(x - x1 * grad, r)
    fx2 = f(x - x2 * grad, r)
    evals = 2
    while abs(b - a) > tol:
        if fx1 < fx2:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = b - phi * (b - a)
            fx1 = f(x - x1 * grad, r)
        else:
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = a + phi * (b - a)
            fx2 = f(x - x2 * grad, r)
        evals += 1
    return (a + b) / 2, evals

def greiciausias_nusileidimas(x0, r, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    total_func_evals = 0
    iter_count = 0

    grad = baudos_funkcijos_gradientas(x, r)
    total_func_evals += 1

    for _ in range(max_iter):
        gamma, evals = auksinio_pjuvio_paieska(baudos_funkcija, x, grad, r)
        total_func_evals += evals

        x_new = x - gamma * grad
        total_func_evals += 2
        old_penalty = baudos_funkcija(x, r)
        new_penalty = baudos_funkcija(x_new, r)

        while new_penalty > old_penalty and gamma > tol:
            gamma *= 0.9
            x_new = x - gamma * grad
            new_penalty = baudos_funkcija(x_new, r)
            total_func_evals += 1

        iter_count += 1

        if np.linalg.norm(grad) < tol or gamma < tol or np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new
        grad = baudos_funkcijos_gradientas(x, r)
        total_func_evals += 1

    return x, iter_count, total_func_evals

def issami_baudos_metodo_seka(x0, r_seka, c=1e-3):
    x = np.array(x0, dtype=float)
    total_iters = 0
    total_func_evals = 0
    print(f"\nPradinis taskas: {np.round(x, 6)}")
    for r in r_seka:
        tol_r = c * r  # tol priklauso nuo r
        x_prev = x.copy()
        x, iters, func_evals = greiciausias_nusileidimas(x, r, tol=tol_r, max_iter=1000)
        total_iters += iters
        total_func_evals += func_evals
        fval = tikslo_funkcija(x)
        gval = gi(x)
        hval = hi(x)
        penalty = baudos_funkcija(x, r)
        print(f"r={r:.5f} | x=({x[0]:.6f}, {x[1]:.6f}, {x[2]:.6f}) | f(x)={-fval:.6f} | g(x)={gval:.6f} | h(x)={[round(h,6) for h in hval]} | Penalty={penalty:.6f} | Iter={iters} | FuncEval={func_evals}")
    return x, total_iters, total_func_evals

def main():
    x0 = [0.0, 0.0, 0.0]
    x1 = [1.0, 1.0, 1.0]
    xm = [0.1, 0.4, 0.5]
    r_seka = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.1, 0.05, 0.01]
    rezultatai = []
    for pradinis in [x0, x1, xm]:
        print("\n"+"="*80)
        galutinis, iter_suma, func_eval_suma = issami_baudos_metodo_seka(pradinis, r_seka)
        fval = -tikslo_funkcija(galutinis)
        gval = gi(galutinis)
        hval = hi(galutinis)
        rezultatai.append((pradinis, galutinis, fval, gval, hval, iter_suma, func_eval_suma))
    print("\n"+"="*80)
    lentele = PrettyTable()
    lentele.field_names = [
        "Pradinis taskas", "Galutinis taskas (gauti sprendiniai)", "Tikslo f.", "g(x)", "h(x)", "Iteracijos", "Funkc. kvietimai"
    ]
    for prad, galut, fval, gval, hval, iter_suma, func_eval_suma in rezultatai:
        lentele.add_row([
            f"[{prad[0]:.2f}, {prad[1]:.2f}, {prad[2]:.2f}]",
            f"[{galut[0]:.6f}, {galut[1]:.6f}, {galut[2]:.6f}]",
            f"{-fval:.6f}",
            f"{gval:.6f}",
            f"[{hval[0]:.6f}, {hval[1]:.6f}, {hval[2]:.6f}]",
            iter_suma,
            func_eval_suma
        ])
    print(lentele)

if __name__ == "__main__":
    main()
