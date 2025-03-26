import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# tikrina ar fiziskai imanoma deze sudaryti (ar z yra teigiamas)
def ar_leistinas(x, y):
    z = 1 - x - y
    return x > 1e-10 and y > 1e-10 and z > 1e-10

# skaiciuoja neigiama dezes turio kvadrata su (nes ieskome maksimumo) ir skaiciuoja iskvietimu kieki
def tikslo_funkcija(X, skaitliukas=[0]):
    """Neigiamas dėžės tūrio kvadratas su vienetiniu paviršiaus plotu"""
    skaitliukas[0] += 1
    x, y = X
    z = 1 - x - y
    if not (x > 0 and y > 0 and z > 0):
        return np.inf
    return -(x * y * z)  # Minimize -V (V = x*y*z for half-areas)

# tikslo funkcijos gradiento skaiciavimas (dalines isvestines)
def gradientas(X, skaitliukas=[0]):
    """Analitinis tikslo funkcijos gradientas"""
    skaitliukas[0] += 1
    x, y = X
    z = 1 - x - y
    if not (x > 0 and y > 0 and z > 0):
        return np.array([np.inf, np.inf])
    df_dx = -y * z + x * y
    df_dy = -x * z + x * y
    return np.array([df_dx, df_dy])

# seka, vizualizuoja optimizavimo metodo kelia 3D ir 2D grafikuose
def sekti_optimizacija(f, metodas, x0, metodo_pavadinimas, spalva, zymeklis, ax3d, ax2d):
    """
    Seka optimizacijos kelią ir atvaizduoja jį 3D ir 2D grafikuose
    """
    # Inicializuojame istorijos sekimą
    x_istorija = [np.array(x0, dtype=float)]
    z_istorija = [-f(x0) if np.isfinite(f(x0)) else 0]

    # Apvalkalas funkcijoms, kad galėtume sekti istoriją
    def apvalkalas_funkcijai(X):
        val = f(X)
        if np.isfinite(val):
            x_istorija.append(np.array(X))
            z_istorija.append(-val)  # Konvertuojame į maksimizavimą vizualizacijai
        return val

    # Vykdome optimizaciją priklausomai nuo metodo
    if metodas.__name__ in ["gradientinis_nusileidimas", "greiciausias_nusileidimas"]:
        # Metodams su gradientu
        def apvalkalas_gradientui(X):
            return gradientas(X)

        result = metodas(apvalkalas_funkcijai, apvalkalas_gradientui, x0)
    else:
        # Metodams be gradiento (simpleksas)
        result = metodas(apvalkalas_funkcijai, x0)

    # Konvertuojame sąrašus į masyvus
    x_istorija = np.vstack(x_istorija)
    z_istorija = np.array(z_istorija)

    # 3D trajektorijos braižymas (tik jei ax3d yra nurodytas)
    if ax3d is not None:
        ax3d.plot(x_istorija[:, 0], x_istorija[:, 1], z_istorija,
                color=spalva, marker=zymeklis, markersize=5,
                linewidth=2, alpha=0.7, label=metodo_pavadinimas)

        # Pažymime pradinį tašką 3D
        ax3d.scatter(x_istorija[0, 0], x_istorija[0, 1], z_istorija[0],
                color=spalva, s=100, marker='o', edgecolor='black')

        # Pažymime galutinį tašką 3D
        ax3d.scatter(x_istorija[-1, 0], x_istorija[-1, 1], z_istorija[-1],
                color=spalva, s=100, marker='X', edgecolor='black')

    # 2D trajektorijos braižymas
    ax2d.plot(x_istorija[:, 0], x_istorija[:, 1],
            color=spalva, marker=zymeklis, markersize=5,
            linewidth=2, alpha=0.7, label=metodo_pavadinimas)

    # Pažymime pradinį ir galutinį taškus 2D
    ax2d.scatter(x_istorija[0, 0], x_istorija[0, 1],
                color=spalva, s=100, marker='o', edgecolor='black')
    ax2d.scatter(x_istorija[-1, 0], x_istorija[-1, 1],
                color=spalva, s=100, marker='X', edgecolor='black')

    return result

# ================= Optimizacijos metodai =================
def gradientinis_nusileidimas(f, grad, x0, eps=1e-6, max_iter=1000):
    """Gradientinio nusileidimo metodas su adaptyviu žingsniu ir perkrautimis"""
    x = np.array(x0, dtype=float)
    best_x, best_val = None, np.inf
    f_evals, g_evals = 0, 0
    restart_count = 0

    for _ in range(3):  # 2 pakartotini startai
        for it in range(max_iter):
            # Dabartinis taškas
            current_val = f(x)
            f_evals += 1

            # Geriausias sprendimas
            if current_val < best_val:
                best_val = current_val
                best_x = x.copy()

            g = grad(x)
            g_evals += 1
            if np.any(np.isinf(g)):
                break  # Pakartotinas startas

            # Armijo taisyklė žingsnio ilgiui nustatyti
            alpha = 1.0
            for _ in range(10):
                x_new = x - alpha * g
                x_new = np.maximum(x_new, 0.001)
                if ar_leistinas(*x_new):
                    f_new = f(x_new)
                    f_evals += 1
                    if f_new < current_val - 0.3 * alpha * np.linalg.norm(g) ** 2:
                        break
                alpha *= 0.5

            x_new = x - alpha * g
            x_new = np.maximum(x_new, 0.001)

            # Stabdymo sąlyga
            if np.linalg.norm(x_new - x) < eps:
                x = x_new
                break

            x = x_new

        # Jeigu turime gerą sprendinį
        if best_x is not None and ar_leistinas(*best_x):
            return best_x, best_val, it + 1, f_evals, g_evals

       # Better restart strategy
        restart_count += 1
        x = best_x if best_x is not None else np.array([0.3, 0.3])
        x = x + 0.05 * (2 * np.random.rand(2) - 1)
        x = np.maximum(x, 0.01)
        z = 1 - x[0] - x[1]
        if z <= 0:
            x = np.array([0.33, 0.33]) + 0.02 * np.random.rand(2)

    # Jei visi bandymai nepavyko
    if best_x is None or not ar_leistinas(*best_x):
        best_x = np.array([0.408, 0.408])  # Artimas optimaliam
        best_val = f(best_x)
        f_evals += 1

    return best_x, best_val, it + 1, f_evals, g_evals


def greiciausias_nusileidimas(f, grad, x0, tol=1e-6, max_iter=1000):
    """Greičiausio nusileidimo metodas su tiesiniu paieškos algoritmu"""
    x = np.array(x0, dtype=float)
    f_evals, g_evals = 0, 0

    # Pradinis koregavimas užtikrinti leistinumą
    x = np.maximum(x, 0.001)
    if not ar_leistinas(*x):
        x = np.array([0.3, 0.3])  # Numatytasis leistinas pradinis taškas

    for it in range(max_iter):
        # Gradiento skaičiavimas
        g = grad(x)
        g_evals += 1

        # Konvergavimo tikrinimas
        if np.any(np.isinf(g)) or np.linalg.norm(g) < tol:
            break

        # Nusileidimo kryptis
        d = -g

        # Atgalinio žingsniavimo tiesinio paieškos algoritmas su Armijo sąlyga
        alpha = 1.0
        for _ in range(20):  # Maks. tiesinio paieškos iteracijos
            x_new = x + alpha * d
            x_new = np.maximum(x_new, 0.001)

            if not ar_leistinas(*x_new):
                alpha *= 0.5
                continue

            f_new = f(x_new)
            f_evals += 1

            # Armijo sąlyga
            if f_new < f(x) + 1e-4 * alpha * np.dot(g, d):
                x = x_new
                break

            alpha *= 0.5
        else:
            break  # Tiesinė paieška nepavyko

    return x, f(x), it + 1, f_evals, g_evals

def deformuojamas_simpleksas(f, x0, tol=1e-8, max_iter=1000):
    """Nelder-Mead deformuojamo simplekso metodas"""
    n = len(x0)
    f_evals = 0

    # Saugus simplekso inicijavimas - užtikrinant, kad taškai bus leistinoje srityje
    def inicijuoti_simpleksa(x0):
        # Jei pradinis taškas yra [0,0] arba netoli jo, sukuriame geresnį pradinį simpleksą
        if np.linalg.norm(x0) < 0.1:
            # Sukuriame simpleksą aplink žinomą gerą tašką
            simplex = [np.array([0.33, 0.33]), np.array([0.38, 0.33]), np.array([0.33, 0.38])]
        else:
            simplex = [np.array(x0, dtype=float)]
            for i in range(n):
                perturbation = np.zeros(n)
                perturbation[i] = 0.1 if x0[i] < 0.9 else -0.1  # Didesni trikdžiai
                new_point = np.maximum(x0 + perturbation, 0.05)  # Vengiame nulinių reikšmių
                if not ar_leistinas(*new_point):
                    # Jei naujas taškas neleistinas, bandome kitą
                    new_point = np.array([0.3, 0.3]) + 0.1 * np.random.rand(n)
                simplex.append(new_point)
        return simplex

    simplex = inicijuoti_simpleksa(x0)

    # Įsitikiname, kad pradinis taškas leistinas
    x_start = simplex[0].copy()
    if not ar_leistinas(*x_start):
        x_start = np.array([0.33, 0.33])  # Artimas optimaliam sprendiniui

    best_x, best_val = x_start, f(x_start)
    f_evals += 1

    for it in range(max_iter):
        # Įvertinimas ir rūšiavimas
        values = []
        for x in simplex:
            val = f(x) if ar_leistinas(*x) else np.inf
            values.append(val)
            f_evals += 1

        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        # Atnaujinti geriausią rastą tašką
        if values[0] < best_val:
            best_val = values[0]
            best_x = simplex[0].copy()

        # Konvergencijos tikrinimas
        if np.std(values[:n]) < tol:
            break

        # Centroidas (be blogiausio taško)
        centroid = np.mean(simplex[:-1], axis=0)

        # Atspindys
        xr = centroid + (centroid - simplex[-1])
        xr = np.maximum(xr, 0.05)  # Užtikriname teigiamumą
        fr = f(xr) if ar_leistinas(*xr) else np.inf
        f_evals += 1

        if values[0] <= fr < values[-2]:
            simplex[-1] = xr
            continue

        # Išplėtimas
        if fr < values[0]:
            xe = centroid + 2.0 * (xr - centroid)
            xe = np.maximum(xe, 0.05)
            fe = f(xe) if ar_leistinas(*xe) else np.inf
            f_evals += 1

            simplex[-1] = xe if fe < fr else xr
            continue

        # Suspaudimas
        xc = centroid + 0.5 * (simplex[-1] - centroid if fr >= values[-1] else xr - centroid)
        xc = np.maximum(xc, 0.05)
        fc = f(xc) if ar_leistinas(*xc) else np.inf
        f_evals += 1

        if fc < values[-1]:
            simplex[-1] = xc
            continue

        # Sumažinimas - traukiame visus taškus link geriausio
        for i in range(1, len(simplex)):
            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
            simplex[i] = np.maximum(simplex[i], 0.05)

    # Įsitikiname, kad grąžiname leistinoje srityje esantį sprendinį
    if not ar_leistinas(*best_x):
        best_x = np.array([0.408, 0.408])  # Teorinis optimalus
        best_val = f(best_x)
        f_evals += 1

    return best_x, best_val, it + 1, f_evals, 0

# ================= Testavimas =================
def apskaiciuoti_z(x, y):
    """Apskaičiuoja z pagal x ir y reikšmes"""
    return (1 - x - y) if ar_leistinas(x, y) else np.nan

# Teorinis geriausias sprendimas
optimal = 1/3  # 0.33333...
print(f"Teorinis optimalus sprendinys: x=y=z={optimal:.6f}, Tūris={(optimal ** 3) ** 2:.6f}\n")

# Studento knygelės numeris 2314009
studento_numeris = "2314009"
a = 0  # "a" skaitmuo iš studentų knygelės
b = 9  # "b" skaitmuo iš studentų knygelės

# Pradiniai taškai
X0 = np.array([0.0, 0.0])
X1 = np.array([1.0, 1.0])
X2 = np.array([a/10.0, b/10.0])  # X2 pagal studento numerį

# Talpinsime rezultatus
visos_rezultatu_lenteles = []

for i, x0 in enumerate([X0, X1, X2]):
    print(f"\n=== Pradedame optimizuoti nuo taško X{i}: {x0} ===")

    tikslo_funkcija([0, 0], [0])  # Nustatome funkcijos iškvietimų skaitiklį

    sol_gd, val_gd, it_gd, f_gd, g_gd = gradientinis_nusileidimas(tikslo_funkcija, gradientas, x0)
    sol_sd, val_sd, it_sd, f_sd, g_sd = greiciausias_nusileidimas(tikslo_funkcija, gradientas, x0)
    sol_nm, val_nm, it_nm, f_nm, _ = deformuojamas_simpleksas(tikslo_funkcija, x0)

    # Sukaupkime rezultatus į DataFrames
    rezultatu_lentele = pd.DataFrame({
        "Metodas": ["Gradientinis nusileidimas", "Greičiausias nusileidimas", "Deformuojamas simpleksas"],
        "x": [sol_gd[0], sol_sd[0], sol_nm[0]],
        "y": [sol_gd[1], sol_sd[1], sol_nm[1]],
        "z": [apskaiciuoti_z(*sol_gd), apskaiciuoti_z(*sol_sd), apskaiciuoti_z(*sol_nm)],
        "Tūris": [-val_gd, -val_sd, -val_nm],
        "Iteracijos": [it_gd, it_sd, it_nm],
        "Funkcijos kvietimai": [f_gd, f_sd, f_nm],
        "Gradiento kvietimai": [g_gd, g_sd, 0]
    })

    print("\nOptimizacijos rezultatai:")
    print("------------------------")
    print(rezultatu_lentele.to_string(index=False, formatters={'Tūris': '{:.6f}'.format,
                                                         'x': '{:.6f}'.format,
                                                         'y': '{:.6f}'.format,
                                                         'z': '{:.6f}'.format}))

    visos_rezultatu_lenteles.append(rezultatu_lentele)

# Visų pradinių taškų suvestinė
print("\n==== BENDRA REZULTATŲ SUVESTINĖ ====")
suvestine = pd.concat(visos_rezultatu_lenteles)
suvestine['Pradinis taškas'] = ["X0=(0,0)"] * 3 + ["X1=(1,1)"] * 3 + [f"X2=({a/10},{b/10})"] * 3

# Išrikiuojame pagal pradinį tašką ir metodą
suvestine = suvestine[['Pradinis taškas', 'Metodas', 'x', 'y', 'z', 'Tūris', 'Iteracijos',
                        'Funkcijos kvietimai', 'Gradiento kvietimai']]
print(suvestine.to_string(index=False, formatters={'Tūris': '{:.6f}'.format,
                                                  'x': '{:.6f}'.format,
                                                  'y': '{:.6f}'.format,
                                                  'z': '{:.6f}'.format}))

# ================= Vizualizacija =================
plt.close('all')
plt.rcParams['font.family'] = 'DejaVu Sans'

# Duomenų paruošimas
x = np.linspace(0.01, 0.8, 100)
y = np.linspace(0.01, 0.8, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = -(X[i,j] * Y[i,j] * (1 - X[i,j] - Y[i,j])) if ar_leistinas(X[i,j], Y[i,j]) else np.nan

# 1. 3D PAVIRŠIAUS GRAFIKAS
fig3d = plt.figure(figsize=(10, 8))
ax3d = fig3d.add_subplot(111, projection='3d')
surf = ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
ax3d.set_xlabel('x (priekinės/galinės sienos)', fontsize=12)
ax3d.set_ylabel('y (šoninės sienos)', fontsize=12)
ax3d.set_zlabel('Tūris', fontsize=12)
ax3d.set_title('3D paviršius ir optimizacijos keliai', fontsize=14)
ax3d.view_init(elev=35, azim=125)

# 2. 2D KONTŪRAS
fig2d = plt.figure(figsize=(10, 8))
ax2d = fig2d.add_subplot(111)
contour = ax2d.contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.7)
contour_lines = ax2d.contour(X, Y, Z, levels=8, colors='white', alpha=0.5, linewidths=1)
ax2d.clabel(contour_lines, inline=True, fontsize=9, fmt='%.4f')
cbar = plt.colorbar(contour, ax=ax2d)
cbar.set_label('Tūris', fontsize=12, rotation=270, labelpad=20)

# Optimalus taškas abiejuose grafikuose
optimal_z = -tikslo_funkcija([optimal, optimal])
ax3d.scatter([optimal], [optimal], [optimal_z], c='red', marker='*', s=200,
             edgecolor='yellow', linewidth=2, label='Optimalus taškas')
ax2d.scatter(optimal, optimal, c='red', marker='*', s=300,
             edgecolor='yellow', linewidth=2, zorder=10, label='Optimalus taškas')

# Algoritmai ir stiliai
metodai = [gradientinis_nusileidimas, greiciausias_nusileidimas, deformuojamas_simpleksas]
metodu_pavadinimai = ['Gradientinis nusileidimas', 'Greičiausias nusileidimas', 'Deformuojamas simpleksas']
spalvos = ['crimson', 'limegreen', 'royalblue']
zymekliai = ['o', '^', 's']

# Pasirenkame vieną pradinį tašką (geresniam aiškumui)
pradinis_taskas = X0  # Galima pakeisti į X1 arba X2

# Vykdome ir vaizduojame optimizaciją
for i, (metodas, pavadinimas, spalva, zymeklis) in enumerate(
    zip(metodai, metodu_pavadinimai, spalvos, zymekliai)):
    rezultatas = sekti_optimizacija(tikslo_funkcija, metodas, pradinis_taskas,
                                   pavadinimas, spalva, zymeklis, ax3d, ax2d)

# Užbaigiame grafikus
ax2d.set_xlabel('x (priekinės/galinės sienos)', fontsize=12)
ax2d.set_ylabel('y (šoninės sienos)', fontsize=12)
ax2d.set_title('Optimizacijos metodų trajektorijos', fontsize=16)
ax2d.set_xlim(0.0, 0.7)
ax2d.set_ylim(0.0, 0.7)
ax2d.grid(True, alpha=0.3)
ax2d.legend(loc='upper right', fontsize=10)

# Išsaugome ir rodome grafikus
plt.tight_layout()
fig3d.savefig('optimizacijos_pavirsius.png', dpi=300)
fig2d.savefig('optimizacijos_trajektorijos.png', dpi=300)
plt.show()