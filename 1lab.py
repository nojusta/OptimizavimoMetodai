import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

def tvarkyti_b(b):
    while b == 0 or b > 9:
        digits_sum = sum(int(d) for d in str(b))
        b = digits_sum
    return b

# Tikslo funkcija
def f(x, a, b):
    return ((x**2 - a)**2) / b - 1

# Padalina intervala i dvi dalis ir iesko, kurioje puseje yra zemesnis taskas
def intervalo_dalijimo_pusiau(f, l, r, eps, a, b):
    eval_count = 0
    points = []  # irasyti esamas xm reiksmes
    
    while True:
        L = r - l
        xm = (l + r) / 2.0
        points.append(xm)
        fxm = f(xm, a, b)
        eval_count += 1
        
        x1 = l + L / 4.0
        x2 = r - L / 4.0
        fx1 = f(x1, a, b)
        fx2 = f(x2, a, b)
        eval_count += 2

        if fx1 < fxm:
            r = xm
            xm = x1
        elif fx2 < fxm:
            l = xm
            xm = x2
        else:
            l = x1
            r = x2

        if (r - l) < eps:
            x_opt = (l + r) / 2.0
            points.append(x_opt)
            return x_opt, eval_count, points

def auksinio_pjuvio_algoritmas(f, l, r, eps, a, b):
    tau = (np.sqrt(5) - 1) / 2  # τ koeficientas 0.6180339887498949
    eval_count = 0 
    points = []  # irasyti bandomuosius xm taskus
    while True:
        L = r - l
        if L < eps:
            x_opt = (l + r) / 2.0
            points.append(x_opt)
            return x_opt, eval_count, points

        x1 = r - tau * L
        x2 = l + tau * L
        points.extend([x1, x2])
        fx1 = f(x1, a, b)
        fx2 = f(x2, a, b)
        eval_count += 2

        if fx2 < fx1:
            l = x1
        else:
            r = x2

def niutono_metodas(x0, eps, a, b):
    """
    Naudojamos pirmosios ir antrosios išvestinės:
      f'(x) = 4*x*(x^2 - a)/b
      f''(x) = (12*x^2 - 4*a)/b
    """
    iterations = 0
    points = [x0]
    while True:
        df = 4 * x0 * (x0**2 - a) / b
        d2f = (12 * x0**2 - 4 * a) / b

        x1 = x0 - df / d2f
        iterations += 1
        points.append(x1)

        if abs(x1 - x0) < eps:
            return x1, iterations, points
        x0 = x1

# parametrai gaunami is studento knygeles (cia pvz. a = 4, b = 5)
a = 4
b = 5

# Kvietimai
x_opt1, evals1, points1 = intervalo_dalijimo_pusiau(f, 0, 10, 1e-4, a, b)
x_opt2, evals2, points2 = auksinio_pjuvio_algoritmas(f, 0, 10, 1e-4, a, b)
x_opt3, iterations3, points3 = niutono_metodas(5, 1e-4, a, b)

# Rezultatu lentele
results = pd.DataFrame({
    "Metodas": ["Intervalo dalijimas pusiau", "Auksinis pjūvis", "Niutono metodas"],
    "Minimumo taškas (x)": [x_opt1, x_opt2, x_opt3],
    "Funkcijos reikšmė": [f(x_opt1, a, b), f(x_opt2, a, b), f(x_opt3, a, b)],
    "Funkcijos skaičiavimai": [evals1, evals2, 0],
    "Žingsniai": [len(points1), len(points2), iterations3]
})
print(results)

# Vizualizacija
x = np.linspace(0, 10, 100)  
y = f(x, a, b)

# Isvestine gauti
x_sym = sp.symbols('x')
func = ((x_sym**2 - a)**2) / b - 1
isvestine = sp.diff(func, x_sym)
isvestine = sp.lambdify(x_sym, isvestine)
y_isvestine = isvestine(x)

plt.figure(figsize=(10, 6))
# braizo pagr. funkcija
plt.plot(x, y, color='red', label="f(x)")
# braizo isvestine
plt.plot(x, y_isvestine, color='blue', label="f'(x)") 
# pridedami taskai 
plt.scatter(points1, f(np.array(points1), a, b), color='blue', label="Intervalo dalijimas")
plt.scatter(points2, f(np.array(points2), a, b), color='red', label="Auksinis pjūvis")
plt.scatter(points3, f(np.array(points3), a, b), color='green', label="Niutono metodas")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Optimizavimo metodų bandymo taškai")
plt.grid()
plt.show()    
