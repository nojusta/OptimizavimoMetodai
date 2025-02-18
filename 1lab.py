import numpy as np
import matplotlib.pyplot as plt

# Tikslo funkcija
def f(x, a, b):
    return ((x**2 - a)**2) / b - 1

def intervalo_dalijimo_pusiau(f, l, r, eps, a, b):
    while True:
        L = r - l
        xm = (l + r) / 2.0
        x1 = l + L / 4.0
        x2 = r - L / 4.0
        
        fxm = f(xm, a, b)
        fx1 = f(x1, a, b)
        fx2 = f(x2, a, b)

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
            return (l + r) / 2.0

def auksinio_pjuvio_algoritmas(f, l, r, eps, a, b):
    tau = (np.sqrt(5) - 1) / 2  # koeficientas
    while True:
        L = r - l
        if L < eps:
            return (l + r) / 2.0
        
        x1 = r - tau * L
        x2 = l + tau * L
        fx1 = f(x1, a, b)
        fx2 = f(x2, a, b)

        if fx2 < fx1:
            l = x1
        else:
            r = x2

def niutono_metodas(x0, eps, a, b):
    """
    Niutono metodo realizacija vienmačiam uždaviniui.
    Naudojamos pirmosios ir antrosios išvestinės:
      f'(x) = 4*x*(x^2 - a) / b
      f''(x) = (12*x^2 - 4*a) / b
    """
    while True:
        df = 4*x0*(x0**2 - a) / b
        d2f = (12*x0**2 - 4*a) / b

        x1 = x0 - df / d2f

        if abs(x1 - x0) < eps:
            return x1

        x0 = x1

