import numpy as np

def simpleksas(C, a, b, kintamieji):
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

    funkc_iskvietimai = 0
    iteracija = 0

    while any(matrica[0][j + 1] < 0 for j in range(n)):
        print("\nTarpine matrica:")
        for row in matrica:
            print(" ".join(f"{val:8.2f}" for val in row))

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

        # Update cost row (C) to match the new tableau
        for j in range(n):
            C[j] = matrica[0][j + 1]

        funkc_iskvietimai += 4
        iteracija += 1

    return matrica, funkc_iskvietimai, iteracija

def kintamieji_is_matricos(matrica, kintamieji):
    m = len(matrica)
    n = len(matrica[0])
    epsilon = 1e-8
    X = {}
    for j in range(1, n):
        if abs(matrica[0][j]) > epsilon:
            X[kintamieji[j]] = 0.0
            continue
        ones = 0
        idx = -1
        for i in range(1, m):
            if abs(matrica[i][j] - 1) < epsilon:
                ones += 1
                idx = i
            elif abs(matrica[i][j]) > epsilon:
                break
        if ones == 1:
            X[kintamieji[j]] = matrica[idx][0]
        else:
            X[kintamieji[j]] = 0.0
    return X

def pagrindine():
    a, b, c = 1, 4, 5
    C = [2, -3, 0, -5, 0, 0, 0]
    a_mat = [
        [-1, 1, -1, -1, 1, 0, 0],
        [2, 4, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1]
    ]
    b_vec = [a, b, c]
    kintamieji = ["b", "x1", "x2", "x3", "x4", "s1", "s2", "s3"]

    print("Simplekso optimizavimas:")
    matrica, funkc, iter_ = simpleksas(C[:], a_mat, b_vec, kintamieji)
    print('\n' + '*'*64)
    print("Rezultatai:")
    for row in matrica:
        print(" ".join(f"{val:8.2f}" for val in row))
    X = kintamieji_is_matricos(matrica, kintamieji)
    print("\nKintamieji:")
    for k in kintamieji[1:]:
        print(f"{k} = {X[k]:.2f}")
    print("\nTikslo funkcijos reiksme:")
    print(f"Z = {-matrica[0][0]:.2f}")
    print(f"\n{iter_} iteracijos")

if __name__ == "__main__":
    pagrindine()
