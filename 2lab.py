import numpy
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def gradientas(x, y):
    grad_x = -y / 8 * (2 * x + y - 1)
    grad_y = -x / 8 * (x + 2 * y - 1)
    return grad_x, grad_y


def tikslo_funkcija(x, y):
    return -1 * ((1 - x - y) * x * y) / 8


class FunkcijosApdorojimas:
    def __init__(self, tikroji_funkcija):
        self.__funkcija = tikroji_funkcija
        self.__atmintis = {} # map'as reiksmem isiminti
        self.__skaiciavimo_kiekis = 0 # funkcijos iskvietimai

    def gauti_atminti(self):
        return self.__atmintis

    def nustatyti_atminti(self, atmintis):
        self.__atmintis = atmintis

    def skaiciuoti(self, taskas, naudoti_atminti=True):
        if not naudoti_atminti:
            return self.__funkcija(taskas[0], taskas[1])

        if taskas in self.__atmintis:
            return self.__atmintis[taskas]

        # jei reiksmes nera atmintyje, skaiciuoja ir isimena
        self.__atmintis[taskas] = self.__funkcija(taskas[0], taskas[1])
        self.__skaiciavimo_kiekis += 1
        return self.__atmintis[taskas]

    def gauti_skaiciavimus(self):
        return self.__skaiciavimo_kiekis

def gradientinis_nusileidimas(funkcija, x0y0, zingsnis):
    e = 1e-4 # zingsnio dydis
    iteracijos = 0
    xy = x0y0 # pradinis taskas
    max_iteracijos = 150
    kelias = [xy] # kelias, kuriuo juda algoritmas

    while iteracijos < max_iteracijos:
        iteracijos += 1

        grad = funkcija.skaiciuoti(xy)
        x_laikinas = xy[0] + zingsnis * grad[0]
        y_laikinas = xy[1] + zingsnis * grad[1]
        laikinas_taskas = (x_laikinas, y_laikinas)
        kelias.append(laikinas_taskas)

        # atstumas tarp dabartinio ir naujo taško
        norm = numpy.linalg.norm([xy[0] - laikinas_taskas[0], xy[1] - laikinas_taskas[1]])
        if norm < e: 
            break

        xy = laikinas_taskas 

    return xy, iteracijos, kelias


def auksinio_pjuvio_paieska(funkcija, taskas, kryptis, reziai=(0, 4)):
    golden = (5 ** 0.5 - 1) / 2
    a, b = reziai # pradinis paieskos intervalas
    c = b - golden * (b - a)
    d = a + golden * (b - a)

    tol = 1e-5
    geriausias_gamma = 1.0
    geriausia_reiksme = float('inf')

    while abs(b - a) > tol:
        fc = funkcija.skaiciuoti((taskas[0] + c * kryptis[0], taskas[1] + c * kryptis[1]))
        fd = funkcija.skaiciuoti((taskas[0] + d * kryptis[0], taskas[1] + d * kryptis[1]))

        if fc < geriausia_reiksme:
            geriausia_reiksme = fc
            geriausias_gamma = c
        if fd < geriausia_reiksme:
            geriausia_reiksme = fd
            geriausias_gamma = d

        if fc < fd:
            b = d
            d = c
            c = b - golden * (b - a)
        else:
            a = c
            c = d
            d = a + golden * (b - a)

    return geriausias_gamma


def greiciausias_nusileidimas(funkcija, gradientas, x0y0):
    e = 1e-3
    iteracijos = 0
    xy = x0y0
    max_iteracijos = 50
    kelias = [xy]
    min_grad = 1e-6 # minimalus gradiento dydis

    while iteracijos < max_iteracijos:
        iteracijos += 1
        grad = gradientas.skaiciuoti(xy)

        grad_dydis = numpy.linalg.norm([grad[0], grad[1]])
        if grad_dydis < min_grad:
            break

        if grad_dydis > 1:
            grad = (grad[0] / grad_dydis, grad[1] / grad_dydis)

        gamma = auksinio_pjuvio_paieska(funkcija, xy, grad)
        laikinas_taskas = (xy[0] + gamma * grad[0], xy[1] + gamma * grad[1])
        kelias.append(laikinas_taskas)

        if numpy.linalg.norm(
                (xy[0] - laikinas_taskas[0], xy[1] - laikinas_taskas[1])) < e:
            break

        xy = laikinas_taskas

    return xy, iteracijos, kelias

def deformuojamas_simpleksas(funkcija, x0y0, zingsnis):
    e = 1e-3
    iteracijos = 0
    max_iteracijos = 80
    n = len(x0y0) 
    kelias = [x0y0] 
    visi_simpleksai = []

    simplexas = numpy.zeros((n + 1, n))
    simplexas[0] = numpy.array([x0y0[0], x0y0[1]])

    d1 = (numpy.sqrt(n + 1) + n - 1) / (n * numpy.sqrt(2)) * zingsnis
    d2 = (numpy.sqrt(n + 1) - 1) / (n * numpy.sqrt(2)) * zingsnis

    for i in range(1, n + 1):
        for j in range(n):
            if i == j + 1:
                simplexas[i][j] = simplexas[0][j] + d2
            else:
                simplexas[i][j] = simplexas[0][j] + d1

    visi_simpleksai.append(numpy.copy(simplexas))

    while iteracijos < max_iteracijos:
        iteracijos += 1
        blogiausia = 0
        geresne = 0

        for i in range(1, n + 1):
            if funkcija.skaiciuoti((simplexas[i][0], simplexas[i][1])) > funkcija.skaiciuoti(
                    (simplexas[blogiausia][0], simplexas[blogiausia][1])):
                blogiausia = i

            if funkcija.skaiciuoti((simplexas[i][0], simplexas[i][1])) < funkcija.skaiciuoti(
                    (simplexas[geresne][0], simplexas[geresne][1])):
                geresne = i

        # Skaiciuojame centroida
        xc = numpy.zeros(n)
        for i in range(n + 1):
            if i != blogiausia:
                xc += simplexas[i]
        xc /= n
        
        # blogiausios virsunes atspindejimas
        xr = -simplexas[blogiausia] + 2 * xc 

        # Jei atspindeta virsune geresne nei blogiausia
        if funkcija.skaiciuoti((xr[0], xr[1])) < funkcija.skaiciuoti(
                (simplexas[blogiausia][0], simplexas[blogiausia][1])):
            simplexas[blogiausia] = xr
            visi_simpleksai.append(numpy.copy(simplexas))
        else:
            # Sutraukimas
            xc = (simplexas[blogiausia] + xc) / 2 

            # Sumažinimai
            for i in range(n + 1):
                if i != blogiausia:
                    simplexas[i] = (simplexas[i] + simplexas[blogiausia]) / 2

                if funkcija.skaiciuoti((simplexas[i][0], simplexas[i][1])) < funkcija.skaiciuoti(
                        (simplexas[geresne][0], simplexas[geresne][1])):
                    geresne = i
            
            visi_simpleksai.append(numpy.copy(simplexas))

            if funkcija.skaiciuoti((xr[0], xr[1])) < funkcija.skaiciuoti(
                    (simplexas[geresne][0], simplexas[geresne][1])):
                simplexas[blogiausia] = xr
                visi_simpleksai.append(numpy.copy(simplexas))
            else:
                for i in range(n + 1):
                    if i != geresne:
                        simplexas[i] = (simplexas[i] + simplexas[geresne]) / 2

                    if funkcija.skaiciuoti((simplexas[i][0], simplexas[i][1])) < funkcija.skaiciuoti(
                            (simplexas[geresne][0], simplexas[geresne][1])):
                        geresne = i
                visi_simpleksai.append(numpy.copy(simplexas))

        # Sustojimo kriterijus
        if numpy.linalg.norm(simplexas[blogiausia] - simplexas[geresne]) < e:
            break

        kelias.append((simplexas[geresne][0], simplexas[geresne][1]))

    return (simplexas[geresne][0], simplexas[geresne][1]), iteracijos, kelias, visi_simpleksai

def rezultatu_braizymas(rezultatai):
    sugrupuoti_rezultatai = {}
    for rezultatas in rezultatai:
        if rezultatas['algoritmas'] not in sugrupuoti_rezultatai:
            sugrupuoti_rezultatai[rezultatas['algoritmas']] = []
        sugrupuoti_rezultatai[rezultatas['algoritmas']].append(rezultatas)

    for algoritmas, alg_rezultatai in sugrupuoti_rezultatai.items():
        fig = plt.figure(figsize=(15, 6))
        fig.suptitle(f"Optimizavimo rezultatai: {algoritmas}", fontsize=18, y=0.98)

        for idx, rezultatas in enumerate(alg_rezultatai):
            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
            x_reiksmes = (-1, 1)
            y_reiksmes = (-1, 1)
            xs = numpy.linspace(x_reiksmes[0], x_reiksmes[1], 100)
            ys = numpy.linspace(y_reiksmes[0], y_reiksmes[1], 100)
            xs, ys = numpy.meshgrid(xs, ys)
            zs = numpy.array([rezultatas['braizymo_funkcija'].skaiciuoti((x, y), False) for x, y in zip(xs, ys)])

            ax.plot_surface(xs, ys, zs, cmap='plasma', alpha=0.8)

            if rezultatas['algoritmas'] != 'Deformuojamo simplekso':
                kelias = rezultatas['kelias']
                kelias_x = [p[0] for p in kelias]
                kelias_y = [p[1] for p in kelias]
                kelias_z = [tikslo_funkcija(x, y) for x, y in kelias]
                
                ax.scatter(kelias_x, kelias_y, kelias_z, color='red', marker='o', s=50)
                ax.plot(kelias_x, kelias_y, kelias_z, color='blue', linewidth=2, linestyle='--')
            
            else:
                simpleksai = rezultatas['visi_simpleksai']
                
                for i, simpleksas in enumerate(simpleksai):
                    virsunes_x = [simpleksas[0][0], simpleksas[1][0], simpleksas[2][0], simpleksas[0][0]]
                    virsunes_y = [simpleksas[0][1], simpleksas[1][1], simpleksas[2][1], simpleksas[0][1]]
                    virsunes_z = [tikslo_funkcija(x, y) for x, y in zip(virsunes_x, virsunes_y)]
                    
                    alpha = max(0.2, 1.0 - i * 0.8 / len(simpleksai))
                    
                    ax.plot(virsunes_x, virsunes_y, virsunes_z, 'b-', linewidth=1.5*alpha)
                    
                    if i == 0 or i == len(simpleksai)-1:
                        ax.scatter(virsunes_x[:3], virsunes_y[:3], virsunes_z[:3], 
                                  color='red' if i == 0 else 'red', s=30)

            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.set_zlabel('Z', fontsize=10)
            ax.set_title(f"Pradinis taškas: ({rezultatas['kelias'][0][0]:.2f}, {rezultatas['kelias'][0][1]:.2f})", fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)

def main():
    # Pradiniai parametrai
    A = 4
    B = 5
    pradines_reiksmes = [[0, 0], [1, 1], [A / 10, B / 10]]

    funkcijos = [FunkcijosApdorojimas(tikslo_funkcija) for _ in range(3)]
    gradiento_funkcijos = [FunkcijosApdorojimas(gradientas) for _ in range(2)]

    rezultatai = []

    # Gradientinio nusileidimo
    for taskas in pradines_reiksmes:
        ats, iteracijos, kelias = gradientinis_nusileidimas(gradiento_funkcijos[0], (taskas[0], taskas[1]), 0.9)
        rezultatai.append({
            'taskas': taskas,
            'algoritmas': 'Gradientinio nusileidimo',
            'atsakymas': ats,
            'f_reiksme': tikslo_funkcija(ats[0], ats[1]),
            'iteracijos': iteracijos,
            'kvietimai': gradiento_funkcijos[0].gauti_skaiciavimus(),
            'kvietimu_tipas': 'Gradiento',
            'braizymo_funkcija': funkcijos[0],
            'kelias': kelias
        })

    # Greiciausio nusileidimo
    for taskas in pradines_reiksmes:
        ats, iteracijos, kelias = greiciausias_nusileidimas(funkcijos[1], gradiento_funkcijos[1],
                                                            (taskas[0], taskas[1]))
        rezultatai.append({
            'taskas': taskas,
            'algoritmas': 'Greiciausio nusileidimo',
            'atsakymas': ats,
            'f_reiksme': tikslo_funkcija(ats[0], ats[1]),
            'iteracijos': iteracijos,
            'kvietimai': gradiento_funkcijos[1].gauti_skaiciavimus(),
            'kvietimu_tipas': 'Gradiento',
            'braizymo_funkcija': funkcijos[1],
            'kelias': kelias
        })

    # Deformuojamo simplekso
    for taskas in pradines_reiksmes:
        ats, iteracijos, kelias, visi_simpleksai = deformuojamas_simpleksas(funkcijos[2], (taskas[0], taskas[1]), 0.3)
        rezultatai.append({
            'taskas': taskas,
            'algoritmas': 'Deformuojamo simplekso',
            'atsakymas': ats,
            'f_reiksme': tikslo_funkcija(ats[0], ats[1]),
            'iteracijos': iteracijos,
            'kvietimai': funkcijos[2].gauti_skaiciavimus(),
            'kvietimu_tipas': 'Funkcijos',
            'braizymo_funkcija': funkcijos[2],
            'kelias': kelias,
            'visi_simpleksai': visi_simpleksai  
        })

    # Sukuria lentele
    table = PrettyTable()
    table.field_names = ["Taskas", "Algoritmas", "Rastas minimumas", "Funkcijos reikšmė", "Iteracijų skaičius", "Kvietimai"]

    for rezultatas in rezultatai:
        x, y = rezultatas['atsakymas']
        min_str = f"({x:.6f}, {y:.6f})"
        f_reiksme = f"{rezultatas['f_reiksme']:.6f}"
        iteracijos = rezultatas['iteracijos']
        kvietimai = f"{rezultatas['kvietimai']} ({rezultatas['kvietimu_tipas']})"

        taskas_str = f"({rezultatas['taskas'][0]:.2f}, {rezultatas['taskas'][1]:.2f})"

        table.add_row([
            taskas_str,
            rezultatas['algoritmas'],
            min_str,
            f_reiksme,
            iteracijos,
            kvietimai
        ])

    print("\nOPTIMIZAVIMO REZULTATAI\n")
    print(table)

    rezultatu_braizymas(rezultatai)
    plt.show(block=True)


if __name__ == "__main__":
    main()
