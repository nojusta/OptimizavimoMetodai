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
    grad_h = np.array([
        2*h[0] if h[0] > 0 else 0,
        2*h[1] if h[1] > 0 else 0,
        2*h[2] if h[2] > 0 else 0
    ])
    return grad_f + 1.0/r * (grad_g + grad_h)

def tenkina_apribojimus(x):
    return gi(x) <= 0 and all(xi >= 0 for xi in x)

def auksinio_pjuvio_paieska(f, x, grad, r, a=0, b=10, tol=1e-6):
    phi = (np.sqrt(5) - 1) / 2
    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)
    fx1 = f(x - x1 * grad, r)
    fx2 = f(x - x2 * grad, r)
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
    return (a + b) / 2

def greiciausias_nusileidimas(x0, r, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    i = 0
    for i in range(max_iter):
        grad = baudos_funkcijos_gradientas(x, r)
        if np.linalg.norm(grad) < tol:
            break
        gamma = auksinio_pjuvio_paieska(baudos_funkcija, x, grad, r)
        x_naujas = x - gamma * grad
        sena_bauda = baudos_funkcija(x, r)
        nauja_bauda = baudos_funkcija(x_naujas, r)
        # If penalty increases or constraints violated, reduce gamma
        while (nauja_bauda > sena_bauda or not tenkina_apribojimus(x_naujas)) and gamma > tol:
            gamma *= 0.9
            x_naujas = x - gamma * grad
            nauja_bauda = baudos_funkcija(x_naujas, r)
        if np.linalg.norm(x_naujas - x) < tol or gamma < tol:
            x = x_naujas
            break
        x = x_naujas
    return x, i+1 if i > 0 else 1

def baudos_metodo_seka(x0, r_seka, tol=1e-6):
    x = np.array(x0, dtype=float)
    iteraciju_suma = 0
    for r in r_seka:
        x_prev = x.copy()
        x, iteracijos = greiciausias_nusileidimas(x, r, tol)
        iteraciju_suma += iteracijos
        if np.linalg.norm(x - x_prev) < tol:
            break
    return x, iteraciju_suma

def spausdinti_lentele(rezultatai):
    lentele = PrettyTable()
    lentele.field_names = ["Pradinis taskas", "Galutinis taskas", "Tikslo f.", "Iteraciju suma"]
    for prad, galut, fval, iter_suma in rezultatai:
        lentele.add_row([
            f"[{prad[0]:.2f}, {prad[1]:.2f}, {prad[2]:.2f}]",
            f"[{galut[0]:.4f}, {galut[1]:.4f}, {galut[2]:.4f}]",
            f"{-fval:.6f}",
            iter_suma
        ])
    print(lentele)

def main():
    x0 = [0.0, 0.0, 0.0]
    x1 = [1.0, 1.0, 1.0]
    xm = [0.1, 0.4, 0.5]
    r_seka = [10, 5, 3, 2, 1, 0.5, 0.1, 0.001]
    rezultatai = []
    for pradinis in [x0, x1, xm]:
        galutinis, iter_suma = baudos_metodo_seka(pradinis, r_seka)
        fval = tikslo_funkcija(galutinis)
        rezultatai.append((pradinis, galutinis, fval, iter_suma))
    spausdinti_lentele(rezultatai)
    print("\nTeorinis optimalus taskas: x1 = x2 = x3 = 1/3, turis = 0.037037")

if __name__ == "__main__":
    main()
