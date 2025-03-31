import numpy
import matplotlib.pyplot as plt
from tabulate import tabulate

def gradientas(x, y):
    grad_x = -y/8*(2*x+y-1)
    grad_y = -x/8*(x+2*y-1)
    return grad_x, grad_y

# Gradientinio nusileidimo algoritmas
def GradNusileidimas(gradientas, x0y0, zingsnis):
    e = 1e-4 # e - tolerancija, kuri nurodo, kada sustoti
    iteracijos = 0
    xy = x0y0 # xy - dabartinė taško pozicija
    max_iter = 100
    path = [xy]  # Sekti kelią

    while iteracijos < max_iter:
        iteracijos += 1

        z = gradientas.kviesti(xy)
        x_laikinas = xy[0] + zingsnis * z[0]
        y_laikinas = xy[1] + zingsnis * z[1]
        xy_laikinas = (x_laikinas, y_laikinas)
        path.append(xy_laikinas)

        norm = numpy.linalg.norm([xy[0] - xy_laikinas[0], xy[1] - xy_laikinas[1]])
        if norm < e:
            break

        xy = xy_laikinas

    return xy, iteracijos, path  # Grąžina galutinį tašką, iteracijų skaičių ir kelią

# Linijinės paieškos funkcija
def line_search(funkcija, xy, grad, bounds=(0, 4)):
    golden = (5**0.5 - 1) / 2
    a, b = bounds
    c = b - golden * (b - a)
    d = a + golden * (b - a)

    tol = 1e-5
    best_gamma = 1.0
    best_value = float('inf')

    while abs(b - a) > tol:
        fc = funkcija.kviesti((xy[0] + c * grad[0], xy[1] + c * grad[1]))
        fd = funkcija.kviesti((xy[0] + d * grad[0], xy[1] + d * grad[1]))

        # Sekti geriausią rastą reikšmę
        if fc < best_value:
            best_value = fc
            best_gamma = c
        if fd < best_value:
            best_value = fd
            best_gamma = d

        if fc < fd:
            b = d
            d = c
            c = b - golden * (b - a)
        else:
            a = c
            c = d
            d = a + golden * (b - a)

    return best_gamma  # Grąžina geriausią rastą žingsnio dydį

# Greičiausio nusileidimo algoritmas
def GrcNusileid(funkcija, funkcijos_gradientas, x0y0):
    e = 1e-3
    iteracijos = 0
    # xy - dabartinė taško pozicija
    xy = x0y0
    max_iter = 50
    path = [xy] # path - saugo visą taškų kelią
    min_grad = 1e-6

    while iteracijos < max_iter:
        iteracijos += 1
        grad = funkcijos_gradientas.kviesti(xy)

        grad_magnitude = numpy.linalg.norm([grad[0], grad[1]])
        if grad_magnitude < min_grad:
            break

        if grad_magnitude > 1:
            grad = (grad[0]/grad_magnitude, grad[1]/grad_magnitude)

        gamma = line_search(funkcija, xy, grad)
        xy_laikinas = (xy[0] + gamma * grad[0], xy[1] + gamma * grad[1])
        path.append(xy_laikinas)

        if numpy.linalg.norm((xy[0] - xy_laikinas[0], xy[1] - xy_laikinas[1])) < e:
            break

        xy = xy_laikinas

    return xy, iteracijos, path

# Deformuojamo simplekso algoritmas
def DefSimplex(funkcija, x0y0, a):
    e = 1e-3
    iteracijos = 0
    max_iter = 80
    n = len(x0y0)

    simplexas = numpy.zeros((n + 1, n))
    simplexas[0] = numpy.array([ x0y0[0], x0y0[1] ])

    d1 = (numpy.sqrt(n + 1) + n - 1) / (n * numpy.sqrt(2)) * a
    d2 = (numpy.sqrt(n + 1) - 1) / (n * numpy.sqrt(2)) * a

    for i in range(1, n + 1):
        for j in range(n):
            if i == j + 1:
                simplexas[i][j] = simplexas[0][j] + d2
            else:
                simplexas[i][j] = simplexas[0][j] + d1

    while iteracijos < max_iter:
        iteracijos += 1
        blogiausia = 0
        geresne = 0

        for i in range(1, n + 1):
            if funkcija.kviesti((simplexas[i][0], simplexas[i][1])) > funkcija.kviesti((simplexas[blogiausia][0], simplexas[blogiausia][1])):
                blogiausia = i

            if funkcija.kviesti((simplexas[i][0], simplexas[i][1])) < funkcija.kviesti((simplexas[geresne][0], simplexas[geresne][1])):
                geresne = i

        xc = numpy.zeros(n)

        for i in range(n + 1):
            if i != blogiausia:
                xc += simplexas[i]

        xc /= n
        xr = -simplexas[blogiausia] + 2 * xc

        if funkcija.kviesti((xr[0], xr[1])) < funkcija.kviesti((simplexas[blogiausia][0], simplexas[blogiausia][1])):
            simplexas[blogiausia] = xr
        else:
            xc = (simplexas[blogiausia] + xc) / 2

            for i in range(n + 1):
                if i != blogiausia:
                    simplexas[i] = (simplexas[i] + simplexas[blogiausia]) / 2

                if funkcija.kviesti((simplexas[i][0], simplexas[i][1])) < funkcija.kviesti((simplexas[geresne][0], simplexas[geresne][1])):
                    geresne = i

            if funkcija.kviesti((xr[0], xr[1])) < funkcija.kviesti((simplexas[geresne][0], simplexas[geresne][1])):
                simplexas[blogiausia] = xr
            else:
                for i in range(n + 1):
                    if i != geresne:
                        simplexas[i] = (simplexas[i] + simplexas[geresne]) / 2

                    if funkcija.kviesti((simplexas[i][0], simplexas[i][1])) < funkcija.kviesti((simplexas[geresne][0], simplexas[geresne][1])):
                        geresne = i

        if numpy.linalg.norm(simplexas[blogiausia] - simplexas[geresne]) < e:
            break

    return (simplexas[geresne][0], simplexas[geresne][1]), iteracijos

# Pradiniai duomenys
A = 4
B = 5
duomenys = [(0, 0), (1, 1), (A / 10, B / 10)]
# Tikslinė funkcija
def TFunc(x, y):
    return -1*((1-x-y)*x*y)/8

# Klasė, skirta talpyklai ir funkcijos kvietimų skaičiavimui
class FloatFunWrapper:
    def __init__(self, tikslofunkcija):
        self.__funkcija = tikslofunkcija
        self.__talpykla = {}
        self.__kvietimai = 0

    def gauti_talpykla(self):
        return self.__talpykla

    def nustatyti_talpykla(self, talpykla):
        self.__talpykla = talpykla

    def kviesti(self, xy, talpykla = True):
        if not talpykla:
            return self.__funkcija(xy[0], xy[1])

        if xy in self.__talpykla:
            return self.__talpykla[xy]

        self.__talpykla[xy] = self.__funkcija(xy[0], xy[1])
        self.__kvietimai += 1
        return self.__talpykla[xy]

    def kvietimai(self):
        return self.__kvietimai

class TupleFunWrapper:
    def __init__(self, tikslofunkcija):
        self.__funkcija = tikslofunkcija
        self.__talpykla = {}
        self.__kvietimai = 0

    def gauti_talpykla(self):
        return self.__talpykla

    def nustatyti_talpykla(self, talpykla):
        self.__talpykla = talpykla

    def kviesti(self, xy, talpykla = True):
        if not talpykla:
            return self.__funkcija(xy[0], xy[1])

        if xy in self.__talpykla:
            return self.__talpykla[xy]

        self.__talpykla[xy] = self.__funkcija(xy[0], xy[1])
        self.__kvietimai += 1
        return self.__talpykla[xy]

    def kvietimai(self):
        return self.__kvietimai

# 3D grafiko braižymo funkcija
def plot3d(funkcija):
    x_reiksmes = (-1, 1)
    y_reiksmes = (-1, 1)
    xs = numpy.linspace(x_reiksmes[0], x_reiksmes[1], 100)
    ys = numpy.linspace(y_reiksmes[0], y_reiksmes[1], 100)
    xs, ys = numpy.meshgrid(xs, ys)
    zs = numpy.array([funkcija.kviesti((x, y), False) for x, y in zip(xs, ys) ])
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    def prideti_taska(ax, x, y, z, color='red', marker='o', size=20, zorder=1):
        ax.scatter(x, y, z, color=color, marker=marker, s=size, zorder=zorder)

    for key in funkcija.gauti_talpykla().keys():
        prideti_taska(ax, key[0], key[1], funkcija.gauti_talpykla()[key])

    ax.plot_surface(xs, ys, zs, color='yellow')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show(block=True)

# Algoritmų objektai ir vykdymas
GradientoNusileidimas = FloatFunWrapper(TFunc)
GradGradientoNusileidimas = TupleFunWrapper(gradientas)

GreiciausiasNusileidimas = FloatFunWrapper(TFunc)
GradGreicNusileidimas = TupleFunWrapper(gradientas)

Simplexas = FloatFunWrapper(TFunc)
GradSimplexas = TupleFunWrapper(gradientas)

# Add this code after defining your algorithm objects but before the table generation code

results = []

# Run algorithms with each starting point
for xy in duomenys:
    # Reset function calls counters
    GradientoNusileidimas = FloatFunWrapper(TFunc)
    GradGradientoNusileidimas = TupleFunWrapper(gradientas)

    # Gradientinio nusileidimo algoritmas
    answer, iterations, path = GradNusileidimas(GradGradientoNusileidimas, xy, 0.3)
    f_value = TFunc(answer[0], answer[1])

    results.append({
        'algorithm': 'Gradientinis nusileidimas',
        'answer': answer,
        'f_value': f_value,
        'iterations': iterations,
        'grad_calls': GradGradientoNusileidimas.kvietimai(),
        'path': path,
        'plot_func': GradientoNusileidimas
    })

    # Reset function calls counters
    GreiciausiasNusileidimas = FloatFunWrapper(TFunc)
    GradGreicNusileidimas = TupleFunWrapper(gradientas)

    # Greičiausio nusileidimo algoritmas
    answer, iterations, path = GrcNusileid(GreiciausiasNusileidimas, GradGreicNusileidimas, xy)
    f_value = TFunc(answer[0], answer[1])

    results.append({
        'algorithm': 'Greičiausias nusileidimas',
        'answer': answer,
        'f_value': f_value,
        'iterations': iterations,
        'func_calls': GreiciausiasNusileidimas.kvietimai(),
        'grad_calls': GradGreicNusileidimas.kvietimai(),
        'path': path,
        'plot_func': GreiciausiasNusileidimas
    })

    # Reset function calls counters
    Simplexas = FloatFunWrapper(TFunc)

    # Deformuojamo simplekso algoritmas - need to store path for this algorithm
    answer, iterations = DefSimplex(Simplexas, xy, 0.5)
    f_value = TFunc(answer[0], answer[1])

    # For Simplex, we need a placeholder path since DefSimplex doesn't return one
    # You might want to modify DefSimplex to track path if needed
    results.append({
        'algorithm': 'Deformuojamas simpleksas',
        'answer': answer,
        'f_value': f_value,
        'iterations': iterations,
        'func_calls': Simplexas.kvietimai(),
        'path': [xy, answer],  # Simplified path for visualization
        'plot_func': Simplexas
    })

# Now the table code should work because results has items
# Create a single neat table
table_data = []
headers = ["Algorithm", "Initial Point", "Result Point", "Function Value",
           "Iterations", "Function Calls", "Gradient Calls"]

# Group results by initial point
for i, initial_point in enumerate(duomenys):
    # Get all results for this initial point
    point_results = [r for r in results if r['path'][0] == initial_point]

    for result in point_results:
        algorithm = result['algorithm']
        answer = f"({result['answer'][0]:.6f}, {result['answer'][1]:.6f})"
        f_value = f"{result['f_value']:.8f}"
        iterations = result['iterations']
        func_calls = result.get('func_calls', '-')
        grad_calls = result.get('grad_calls', '-')

        table_data.append([
            algorithm,
            f"({initial_point[0]}, {initial_point[1]})",
            answer,
            f_value,
            iterations,
            func_calls,
            grad_calls
        ])

# Print the table
print("\nOptimization Results:")
print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".8f"))

# Rezultatų grupavimas pagal algoritmą
grouped_results = {}
for result in results:
    if result['algorithm'] not in grouped_results:
        grouped_results[result['algorithm']] = []
    grouped_results[result['algorithm']].append(result)

# Kiekvieno algoritmo rezultatų braižymas
for algorithm, alg_results in grouped_results.items():
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(algorithm, fontsize=16, y=1.05)

    for idx, result in enumerate(alg_results):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        x_reiksmes = (-1, 1)
        y_reiksmes = (-1, 1)
        xs = numpy.linspace(x_reiksmes[0], x_reiksmes[1], 100)
        ys = numpy.linspace(y_reiksmes[0], y_reiksmes[1], 100)
        xs, ys = numpy.meshgrid(xs, ys)
        zs = numpy.array([result['plot_func'].kviesti((x, y), False) for x, y in zip(xs, ys) ])

        ax.plot_surface(xs, ys, zs, color='yellow', alpha=0.8)

        path = result['path']
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        path_z = [TFunc(x, y) for x, y in path]

        ax.scatter(path_x, path_y, path_z, color='red', marker='o', s=20, zorder=2)

        ax.plot(path_x, path_y, path_z, color='blue', linewidth=1, zorder=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Pradinis taškas: ({duomenys[idx][0]}, {duomenys[idx][1]})")

    plt.tight_layout()
    plt.show(block=False)

plt.show(block=True)  # Palikti visus langus atidarytus