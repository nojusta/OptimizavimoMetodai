import numpy as np
from prettytable import PrettyTable
# import matplotlib.pyplot as plt # Kol kas nenaudosime grafiku

# --- Konstantos ir Pradiniai Duomenys ---
A = 1
B = 4
C = 5
S_MAX = 10 * A + B  # 4
L_MAX = 10 * B + C  # 45

# --- Tikslo Funkcija ir Gradientas ---
def tikslo_funkcija(x):
    """ Skaiciuoja tikslo funkcija (-V). """
    x1, x2, x3 = x
    # Jei bet kuri koordinate yra 0 ar maziau, turis yra 0.
    if any(xi <= 0 for xi in x):
        return 0.0
    return -x1 * x2 * x3

def tikslo_gradientas(x):
    """ Skaiciuoja tikslo funkcijos gradienta. """
    x1, x2, x3 = x
    # Jei arti nulio, gradientas bus artimas nuliui arba neapibreztas.
    # Grazinam nuliu gradienta, jei esame ant ribos ar uz jos (x_i <= 0).
    # Baudos narys gradiente tures dominuoti.
    if any(xi <= 0 for xi in x):
         return np.array([0.0, 0.0, 0.0])
    grad_x1 = -x2 * x3
    grad_x2 = -x1 * x3
    grad_x3 = -x1 * x2
    return np.array([grad_x1, grad_x2, grad_x3])

# --- Apribojimu Funkcijos ir Gradientai (gi(x) <= 0) ---
def g1(x):
    x1, x2, x3 = x
    # Apsauga nuo labai dideliu neigiamu skaiciu, kurie gali sukelti problemu veliau
    if any(xi < -1e6 for xi in x): return 1e18 # Didelis pazeidimas
    try:
        val = 2 * (x1*x2 + x1*x3 + x2*x3) - S_MAX
        return val if np.isfinite(val) else 1e18 # Grazinam didele reiksme, jei NaN/Inf
    except OverflowError:
        return 1e18

def g1_gradientas(x):
    x1, x2, x3 = x
    try:
        grad = np.array([2 * (x2 + x3), 2 * (x1 + x3), 2 * (x1 + x2)])
        return grad if np.all(np.isfinite(grad)) else np.array([0.0, 0.0, 0.0])
    except OverflowError:
        return np.array([0.0, 0.0, 0.0])

def g2(x):
    x1, x2, x3 = x
    try:
        val = 4 * (x1 + x2 + x3) - L_MAX
        return val if np.isfinite(val) else 1e18
    except OverflowError:
        return 1e18

def g2_gradientas(x):
    # Gradientas yra konstantinis, nera overflow pavojaus
    return np.array([4.0, 4.0, 4.0])

def g3(x): return -x[0]
def g3_gradientas(x): return np.array([-1.0, 0.0, 0.0])
def g4(x): return -x[1]
def g4_gradientas(x): return np.array([0.0, -1.0, 0.0])
def g5(x): return -x[2]
def g5_gradientas(x): return np.array([0.0, 0.0, -1.0])

apribojimai = [g1, g2, g3, g4, g5]
apribojimu_gradientai = [g1_gradientas, g2_gradientas, g3_gradientas, g4_gradientas, g5_gradientas]

# --- Baudos Funkcija ir Gradientas (1/r metodas) ---
def baudos_funkcija(x, r):
    """ Skaiciuoja P(x, r) = f(x) + (1/r) * suma(max(0, gi(x))^2). """
    if r < 1e-18: r = 1e-18 # Apsauga nuo dalybos is nulio
    try:
        bauda = 0.0
        for g in apribojimai:
            g_val = g(x)
            # Saugiau naudoti np.maximum del galimu NaN is g(x)
            bauda += np.maximum(0, g_val)**2
        if not np.isfinite(bauda): bauda = 1e18 # Jei bauda NaN/Inf

        f_val = tikslo_funkcija(x)
        if not np.isfinite(f_val): f_val = 1e18 # Jei f(x) NaN/Inf

        penalty_term = (1.0 / r) * bauda
        if not np.isfinite(penalty_term): penalty_term = 1e18

        total_val = f_val + penalty_term
        return total_val if np.isfinite(total_val) else 1e18
    except OverflowError:
        return 1e18

def baudos_funkcijos_gradientas(x, r):
    """ Skaiciuoja baudos funkcijos gradienta. """
    if r < 1e-18: r = 1e-18
    try:
        gradientas_f = tikslo_gradientas(x)
        if not np.all(np.isfinite(gradientas_f)): gradientas_f = np.zeros(3)

        gradientas_baudos = np.zeros_like(x, dtype=float)
        for g, grad_g_func in zip(apribojimai, apribojimu_gradientai):
            g_reiksme = g(x)
            if g_reiksme > 0 and np.isfinite(g_reiksme):
                grad_g = grad_g_func(x)
                if np.all(np.isfinite(grad_g)):
                    term = 2 * g_reiksme * grad_g
                    # Patikrinam ar pridejus nebus NaN/Inf
                    if np.all(np.isfinite(gradientas_baudos + term)):
                        gradientas_baudos += term
                    else: # Jei pridejus bus NaN/Inf, stabdom baudos gradiento skaiciavima
                        gradientas_baudos = np.zeros(3) # Arba kitaip tvarkomes
                        break
            elif not np.isfinite(g_reiksme): # Jei pats apribojimas NaN/Inf
                 gradientas_baudos = np.zeros(3)
                 break

        if not np.all(np.isfinite(gradientas_baudos)): gradientas_baudos = np.zeros(3)

        penalty_grad_term = (1.0 / r) * gradientas_baudos
        if not np.all(np.isfinite(penalty_grad_term)): penalty_grad_term = np.zeros(3)

        total_grad = gradientas_f + penalty_grad_term
        return total_grad if np.all(np.isfinite(total_grad)) else np.zeros(3)
    except OverflowError:
        return np.zeros(3)


# --- Optimizavimo Algoritmai ---
def auksinio_pjuvio_paieska_1d(func, x_k, kryptis, r, intervalas=(0, 1), tol=1e-5):
    """ Supaprastinta auksinio pjuvio paieska. """
    auksinis_pjuvis = (np.sqrt(5) - 1) / 2
    a, b = intervalas

    # Apsauga nuo blogos krypties ar tasko
    if not np.all(np.isfinite(x_k)) or not np.all(np.isfinite(kryptis)):
        return 0.0 # Grazinam 0 zingsni

    c = b - auksinis_pjuvis * (b - a)
    d = a + auksinis_pjuvis * (b - a)

    try:
        fc = func(x_k + c * kryptis, r)
        fd = func(x_k + d * kryptis, r)
    except Exception: # Jei nepavyksta apskaiciuoti pradiniu tasku
        return (a + b) / 2 # Grazinam viduri

    max_iter = 50
    iteracijos = 0
    while abs(b - a) > tol and iteracijos < max_iter:
        iteracijos += 1
        # Patikrinimas del NaN/Inf
        if not np.isfinite(fc) or not np.isfinite(fd):
            # Jei reiksmes neapibreztos, maziname intervala is abieju pusiu
             new_b = d
             new_a = c
             a = new_a
             b = new_b
             if abs(b-a) < tol: break
             c = b - auksinis_pjuvis * (b - a)
             d = a + auksinis_pjuvis * (b - a)
             try:
                 fc = func(x_k + c * kryptis, r)
                 fd = func(x_k + d * kryptis, r)
             except Exception: break
             continue

        if fc < fd:
            b = d
            d = c
            c = b - auksinis_pjuvis * (b - a)
            fd = fc
            try:
                fc = func(x_k + c * kryptis, r)
            except Exception: break
        else:
            a = c
            c = d
            d = a + auksinis_pjuvis * (b - a)
            fc = fd
            try:
                fd = func(x_k + d * kryptis, r)
            except Exception: break
    return (a + b) / 2

def greiciausias_nusileidimas(pradinis_taskas, r, max_iter_vidines=100, tol_vidines=1e-5):
    """ Supaprastintas greiciausio nusileidimo algoritmas. """
    x = np.array(pradinis_taskas, dtype=float)
    if not np.all(np.isfinite(x)): # Jei pradinis taskas blogas
        print("    Klaida: Blogas pradinis taskas vidiniam ciklui.")
        return pradinis_taskas, 0 # Grazinam pradini taska

    iteracija = 0
    for iteracija in range(max_iter_vidines):
        grad = baudos_funkcijos_gradientas(x, r)
        grad_norma = np.linalg.norm(grad)

        # Jei gradientas neapibreztas arba labai mazas, sustojam
        if not np.isfinite(grad_norma) or grad_norma < tol_vidines:
            break

        kryptis = -grad
        # Naudojam fiksuota intervala auksinio pjuvio paieskai
        gamma = auksinio_pjuvio_paieska_1d(baudos_funkcija, x, kryptis, r, intervalas=(0, 1.0))

        # Ribojam gamma, kad nebutu per didelis zingsnis (bet leidziam buti 0)
        gamma = np.clip(gamma, 0, 1.0) # Leidziam 0, ribojam iki 1

        x_naujas = x + gamma * kryptis

        # Patikrinimas ar naujas taskas yra validus
        if not np.all(np.isfinite(x_naujas)):
            # print(f"    Vidine iter. {iteracija+1}: Gautas NaN/Inf taskas. Sustojama.")
            break # Sustojam, jei taskas tampa neapibreztas

        # PASALINTA: x_naujas = np.maximum(x_naujas, 1e-6)

        # Sustojimo salyga pagal tasko pokyti
        if np.linalg.norm(x_naujas - x) < tol_vidines:
            x = x_naujas
            break
        x = x_naujas
    # Jei galutinis taskas blogas, grazinam paskutini gera (pradini)
    if not np.all(np.isfinite(x)):
        return pradinis_taskas, iteracija + 1
    return x, iteracija + 1

# --- Pagrindinis Baudos Metodo Ciklas ---
def baudos_metodas(pradinis_taskas, pradinis_r=1.0, beta=10, max_iter_isorines=10, tol_isorines=1e-4):
    """ Pagrindine baudos metodo funkcija (1/r). """
    x_k = np.array(pradinis_taskas, dtype=float)
    # PASALINTA: x_k[x_k <= 0] = 1e-4
    r_k = pradinis_r
    rezultatai_isoriniai = []

    print("--- Baudos Metodo Vykdymas (1/r versija, supaprastinta) ---")
    # Spausdinam originalu pradini taska
    print(f"Pradinis taskas: {pradinis_taskas}")
    print(f"Pradinis r: {r_k}, Beta (daliklis): {beta}")
    print(f"Isorines tolerancija: {tol_isorines}, Max isoriniu iteraciju: {max_iter_isorines}")
    print("-" * 30)

    # Kintamasis paskutiniam geram taskui saugoti
    paskutinis_geras_x = x_k.copy()

    for k in range(max_iter_isorines):
        x_k_plius_1, vidines_iteracijos = greiciausias_nusileidimas(x_k, r_k)

        # Patikrinimas ar vidinis ciklas grazino validu taska
        if not np.all(np.isfinite(x_k_plius_1)):
            print(f"  Isorine iter. {k+1}: Vidinė optimizacija grąžino NaN/Inf. Naudojamas paskutinis geras taškas.")
            x_k_plius_1 = paskutinis_geras_x # Grizta prie paskutinio gero tasko
            # Galima nutraukti isorini cikla, jei vidinis nuolat nepavyksta
            # break

        # Saugom si taska kaip potencialiai paskutini gera
        paskutinis_geras_x = x_k_plius_1.copy()

        apribojimu_pazeidimai = [max(0, g(x_k_plius_1)) for g in apribojimai]
        # Saugiau skaiciuoti max pazeidima
        valid_pazeidimai = [p for p in apribojimu_pazeidimai if np.isfinite(p)]
        maksimalus_pazeidimas = max(valid_pazeidimai) if valid_pazeidimai else 0.0
        if not np.isfinite(maksimalus_pazeidimas): maksimalus_pazeidimas = float('inf')

        f_reiksme = tikslo_funkcija(x_k_plius_1)
        if not np.isfinite(f_reiksme): f_reiksme = float('nan')

        rezultatai_isoriniai.append({
            'iteracija': k + 1, 'r': r_k, 'taskas': x_k_plius_1.copy(),
            'f_reiksme': f_reiksme, 'vidines_iter': vidines_iteracijos,
            'max_pazeidimas': maksimalus_pazeidimas
        })

        print(f"Isorine iter. {k+1}, r={r_k:.1e}: x=[{x_k_plius_1[0]:.4f}, {x_k_plius_1[1]:.4f}, {x_k_plius_1[2]:.4f}], f={f_reiksme:.4f}, pazeid={maksimalus_pazeidimas:.4f}")

        pokytis_isorinis = np.linalg.norm(x_k_plius_1 - x_k)
        # Tikrinam ar pokytis ir pazeidimas yra apibrezti ir mazi
        if np.isfinite(pokytis_isorinis) and np.isfinite(maksimalus_pazeidimas):
            if pokytis_isorinis < tol_isorines and maksimalus_pazeidimas < tol_isorines:
                print(f"Konvergavo isorineje iteracijoje {k+1}.")
                x_k = x_k_plius_1
                break
        else:
             # Jei pokytis ar pazeidimas NaN/Inf, kazkas blogai, bet tesiam su paskutiniu geru tasku
             print(f"  Isorine iter. {k+1}: Pokytis ({pokytis_isorinis}) arba pazeidimas ({maksimalus_pazeidimas}) neapibreztas.")
             x_k = paskutinis_geras_x # Naudojam paskutini gera taska kitai iteracijai
             # Galima butu ir nutraukti cikla cia
             # break

        x_k = x_k_plius_1
        r_k /= beta
    else:
        print(f"Pasiektas maksimalus isoriniu iteraciju skaicius ({max_iter_isorines}).")
        # Galutinis taskas yra paskutinis rastas x_k (kuris turetu buti paskutinis_geras_x)
        x_k = paskutinis_geras_x


    # Galutinis taskas yra paskutinis x_k is ciklo arba paskutinis geras, jei ciklas baigesi blogai
    return x_k, rezultatai_isoriniai

# --- Rezultatu Spausdinimas ---
def spausdinti_rezultatus(galutinis_taskas, rezultatai_isoriniai):
    """ Spausdina rezultatus. """
    print("\n--- Optimizavimo Rezultatai ---")
    lentele = PrettyTable()
    lentele.field_names = ["Iter.", "r", "x1", "x2", "x3", "f(x)", "Vid. Iter.", "Max Pazeid."]
    lentele.float_format = ".4"
    for res in rezultatai_isoriniai:
        # Saugiau spausdinti NaN, jei reiksmes blogos
        f_val_str = f"{res['f_reiksme']:.4f}" if np.isfinite(res['f_reiksme']) else "nan"
        paz_val_str = f"{res['max_pazeidimas']:.4f}" if np.isfinite(res['max_pazeidimas']) else "inf"
        x_vals = [f"{v:.4f}" if np.isfinite(v) else "nan" for v in res['taskas']]

        lentele.add_row([
            res['iteracija'], f"{res['r']:.1e}", x_vals[0], x_vals[1],
            x_vals[2], f_val_str, res['vidines_iter'], paz_val_str
        ])
    print(lentele)

    print("\nGalutinis rastas taskas:")
    if np.all(np.isfinite(galutinis_taskas)):
        print(f"  x = [{galutinis_taskas[0]:.6f}, {galutinis_taskas[1]:.6f}, {galutinis_taskas[2]:.6f}]")
        galutine_f_reiksme = tikslo_funkcija(galutinis_taskas)
        print(f"  f(x) = {galutine_f_reiksme:.6f} (Turis V = {-galutine_f_reiksme:.6f})")
        print("\nApribojimu reiksmes galutiniame taske:")
        try:
            g1_val = g1(galutinis_taskas); g2_val = g2(galutinis_taskas)
            print(f"  Pavirsius: {g1_val + S_MAX:.4f} (<= {S_MAX}) -> g1(x) = {g1_val:.4f}")
            print(f"  Krastines: {g2_val + L_MAX:.4f} (<= {L_MAX}) -> g2(x) = {g2_val:.4f}")
            print(f"  x1={galutinis_taskas[0]:.4f}, x2={galutinis_taskas[1]:.4f}, x3={galutinis_taskas[2]:.4f} (> 0)")
        except Exception as e:
            print(f"  Nepavyko apskaiciuoti apribojimu reiksmiu: {e}")
    else:
        print("  Galutinis taskas neapibreztas (NaN arba Inf).")


# --- Vykdymas ---
if __name__ == "__main__":
    X0 = [0.0, 0.0, 0.0]
    X1 = [1.0, 1.0, 1.0]
    Xm = [A / 10.0, B / 10.0, C / 10.0] # [0.0, 0.4, 0.5]
    pradiniai_taskai_list = [("X0", X0), ("X1", X1), ("Xm", Xm)]

    for pavadinimas, prad_taskas in pradiniai_taskai_list:
        print(f"\n===== Vykdymas pradedant nuo {pavadinimas} = {prad_taskas} =====")
        galutinis_taskas_rastas, rezultatai = baudos_metodas(
            pradinis_taskas=prad_taskas,
            pradinis_r=1.0,       # Pradinis r=1.0
            beta=10,             # r mazinimo daliklis
            max_iter_isorines=10,
            tol_isorines=1e-4
        )
        spausdinti_rezultatus(galutinis_taskas_rastas, rezultatai)
        print(f"===== Vykdymas baigtas pradedant nuo {pavadinimas} =====")