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
    eval_count = 0 # rodo kelis kartus funkcija buvo iskviesta/ivertinta (parodo efektyvuma)
    points = []  # issaugoti iteraciju metu apskaiciuotas xm reiksmes
    L = r - l  # skaiciuoja dabartinio intervalo ilgi
    xm = (l + r) / 2.0  # dabartinio intervalo vidurio taskas
    fxm = f(xm, a, b)
    eval_count += 1
    iterations1 = 0

    while True:
        x1 = l + L / 4.0  # 1/4 intervalo is kaires
        x2 = r - L / 4.0  # 1/4 intervalo is desines
        fx1 = f(x1, a, b)
        fx2 = f(x2, a, b)
        eval_count += 2
        iterations1 += 1

        if fx1 < fxm: # jei kairej pusej f reiksme mazesne
            r = xm # nustatom desiniji krasta i dabartini vidurki
            xm = x1 # naujas vidurkis priskiriamas x1
            fxm = fx1
        elif fx2 < fxm: # jei desinej
            l = xm
            xm = x2
            fxm = fx2
        else: # siauriname intervala tarp x1 ir x2
            l = x1
            r = x2

        L = r - l
        points.extend([xm, x1, x2])
        if L < eps:
            x_opt = (l + r) / 2.0 # skaiciuojam vidurki kaip artimiausia minimuma
            points.append(x_opt)
            return x_opt, eval_count, iterations1, points

def auksinio_pjuvio_algoritmas(f, l, r, eps, a, b):
    tau = (np.sqrt(5) - 1) / 2  # τ koeficientas ~= 0.6180339887498949 - padeda optimaliai mazinti intervala
    eval_count = 0 
    points = []  # list'as irasyti bandomuosius xm taskus
    L = r - l  # skaiciuoja dabartinio intervalo ilgi
    x1 = r - tau * L # taskas, esantis nuo desiniojo krasto, atimant tau*l
    x2 = l + tau * L # taskas, esantis nuo kairiojo krasto, pridedant tau*l
    fx1 = f(x1, a, b)
    fx2 = f(x2, a, b)
    eval_count += 2
    iterations2 = 0

    while True:
        if fx2 < fx1: # jei minimumo sritis labiau paslinkusi i desine
            l = x1 # atnaujiname kairiji krasta
            L = r - l  # skaiciuoja dabartinio intervalo ilgi
            x1 = x2
            fx1 = fx2
            x2 = l + tau * L
            fx2 = f(x2, a, b)
        else:
            r = x2 # priesingu atveju, atnaujiname desiniji krasta
            L = r - l  # skaiciuoja dabartinio intervalo ilgi
            x2 = x1
            fx2 = fx1
            x1 = r - tau * L
            fx1 = f(x1, a, b)
        iterations2 += 1
        eval_count += 1
        L = r - l
        points.extend([x1, x2])
        if L < eps:
            x_opt = (x1 + x2) / 2.0
            points.append(x_opt)
            return x_opt, eval_count, iterations2, points

def niutono_metodas(x0, eps, a, b):
    # Niutono metodas, apskaiciuojant isvestines su sympy
    # f(x) = ((x^2 - a)^2) / b - 1

    x = sp.symbols('x', real=True) # simbolinis kintamasis
    func_expr = ((x**2 - a)**2) / b - 1 # funkcijos israiska (tiksline formule)
    f_prime_expr = sp.diff(func_expr, x) # pirmosios isvestines skaiciavimas
    f_doubleprime_expr = sp.diff(f_prime_expr, x) # antrosios isvestines skaiciavimas

    # paverciame isvestines i paprastas (python suprantamas) funkcijas su lambdify
    f_prime = sp.lambdify(x, f_prime_expr)
    f_doubleprime = sp.lambdify(x, f_doubleprime_expr)

    iterations = 0
    points = [x0] # list'as saugoti spejimus

    while True:
        df = f_prime(x0) # pirmoji isvestine
        d2f = f_doubleprime(x0)  # antroji isvestine

        x1 = x0 - df / d2f # pagal niutono formule
        iterations += 1
        points.append(x1)

        if abs(x1 - x0) < eps:
            return x1, iterations, points
        x0 = x1

# Studento knygeles nr “1*1**ab” 
a = 4
b = 5  

# Patikriname ar reikia tvarkyti b
b = tvarkyti_b(b)
print(f"Naudojami parametrai: a = {a}, b = {b}")


# Kvietimai (1e-4 ~= 0.0001)
x_opt1, evals1, iterations1, points1 = intervalo_dalijimo_pusiau(f, 0, 10, 1e-4, a, b) # intervalas [0, 10]
x_opt2, evals2, iterations2, points2 = auksinio_pjuvio_algoritmas(f, 0, 10, 1e-4, a, b)
x_opt3, iterations3, points3 = niutono_metodas(5, 1e-4, a, b)

# pandas rodymo parinktys
pd.set_option('display.max_columns', None)  # Rodyti visus stulpelius
pd.set_option('display.width', None)        # Nerodyti daugtaskiu
pd.set_option('display.float_format', lambda x: '%.4f' % x)  # 4 skaiciai po kablelio

# Rezultatu lentele
results = pd.DataFrame({
    "Metodas": ["Intervalo dalijimas pusiau", "Auksinis pjūvis", "Niutono metodas"],
    "Minimumo taškas (x)": [x_opt1, x_opt2, x_opt3],
    "Funkcijos reikšmė": [f(x_opt1, a, b), f(x_opt2, a, b), f(x_opt3, a, b)],
    "Funkcijos iškvietimų skaičius": [evals1, evals2, '-'], #
    "Žingsniai (iteracijos)": [iterations1, iterations2, iterations3] # pataisyt aukso pjuvio zingsnius
})
print(results)

# Vizualizacija
x = np.linspace(0, 10, 100) # 100 tasku intervale [0, 10]
y = f(x, a, b) # f(x) reiksme

# Isvestines apskaiciavimas
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
# pridedami taskai (iteraciniai skaiciavimai)
plt.scatter(points1, f(np.array(points1), a, b), color='blue', label="Intervalo dalijimas")
plt.scatter(points2, f(np.array(points2), a, b), color='red', label="Auksinis pjūvis")
plt.scatter(points3, f(np.array(points3), a, b), color='green', label="Niutono metodas")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Optimizavimo metodų bandymo taškai")
plt.grid()
plt.show()
