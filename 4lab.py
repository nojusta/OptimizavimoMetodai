import copy

def spausdinti_matrica(matrica, kintamieji, apribojimai):
    W = 8
    print("\n" + " ".join(f"{v:>{W}}" for v in kintamieji))
    for i, row in enumerate(matrica):
        if i == 0:
            label = "  z |"
        else:
            label = f"{apribojimai[i-1]:>4} |"
        print(label + " ".join(f"{val:>{W}.2f}" for val in row))

def simpleksas(C, a, b, kintamieji, apribojimai):
    m = len(a)
    n = len(C)
    matrica = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]
    matrica[0][0] = 0.0
    for i in range(m):
        matrica[i + 1][0] = b[i]
    for j in range(n):
        matrica[0][j + 1] = C[j]
    for i in range(m):
        for j in range(n):
            matrica[i + 1][j + 1] = a[i][j]

    iteracija = 0

    while any(matrica[0][j + 1] < -1e-9 for j in range(n)):
        print(f"\nTarpinė matrica (iteracija {iteracija + 1}):")
        spausdinti_matrica(matrica, kintamieji, apribojimai)
        pivot_column = min((j + 1 for j in range(n)), key=lambda j: matrica[0][j], default=None)
        if pivot_column is None:
            break
        koef = []
        for i in range(m):
            if matrica[i + 1][pivot_column] > 1e-12:
                koef.append(matrica[i + 1][0] / matrica[i + 1][pivot_column])
            else:
                koef.append(float('inf'))
        if all(x == float('inf') for x in koef):
            break
        pivot_row = koef.index(min(koef)) + 1
        pivot_value = matrica[pivot_row][pivot_column]
        for j in range(n + 1):
            matrica[pivot_row][j] /= pivot_value
        for i in range(m + 1):
            if i == pivot_row:
                continue
            factor = matrica[i][pivot_column]
            for j in range(n + 1):
                matrica[i][j] -= factor * matrica[pivot_row][j]
        for j in range(n):
            C[j] = matrica[0][j + 1]
        iteracija += 1

    return matrica, iteracija

def sprendinio_vektorius_ir_baze(matrica, kintamieji):
    m = len(matrica)
    n = len(matrica[0])
    epsilon = 1e-8
    X = {}
    baze = []
    for j in range(1, n):
        col = [matrica[i][j] for i in range(1, m)]
        ones = sum(abs(x - 1) < epsilon for x in col)
        zeros = sum(abs(x) < epsilon for x in col)
        if ones == 1 and zeros == m - 2:
            idx = [i for i in range(1, m) if abs(matrica[i][j] - 1) < epsilon][0]
            X[kintamieji[j]] = matrica[idx][0]
            baze.append(kintamieji[j])
        else:
            X[kintamieji[j]] = 0.0
    return X, baze

def spausdinti_sprendini(matrica, kintamieji, apribojimai, iteracijos):
    print("\nGalutinė sprendimo matrica:")
    spausdinti_matrica(matrica, kintamieji, apribojimai)
    X, baze = sprendinio_vektorius_ir_baze(matrica, kintamieji)
    print("\nSprendinio vektorius:")
    for k in kintamieji[1:]:
        print(f"{k:>4} = {X[k]:.2f}")
    print(f"\nOptimali bazė: {{{', '.join(baze)}}}")
    print(f"\nTikslo funkcijos reikšmė: Z = {-matrica[0][0]:.2f}")
    print(f"{iteracijos} iteracijos")

def pagrindine():
    a, b, c = 1, 4, 5
    C = [2, -3, 0, -5, 0, 0, 0]
    a_mat = [
        [-1, 1, -1, -1, 1, 0, 0],
        [2, 4, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1]
    ]
    b_vec = [a, b, c]
    kintamieji = [" ", "x1", "x2", "x3", "x4", "s1", "s2", "s3"]
    apribojimai = ["1", "2", "3"]

    print("Simplekso optimizavimas:")
    matrica, iter_ = simpleksas(copy.deepcopy(C), a_mat, b_vec, kintamieji, apribojimai)
    print('\n' + '*'*64)
    print("Rezultatai:")
    spausdinti_sprendini(matrica, kintamieji, apribojimai, iter_)

if __name__ == "__main__":
    pagrindine()
