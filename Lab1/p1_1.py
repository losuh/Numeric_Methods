
import numpy as np

A = np.array([
    [-5, -1, -3, -1],
    [-2,  0,  8, -4],
    [-7, -2,  2, -2],
    [ 2, -4, -4,  4]
], dtype=float)

b = np.array([18, -12, 6, -12], dtype=float)


def lu_decomposition(A):
    n = len(A)
    U = A.tolist()
    L = [[float(i == j) for j in range(n)] for i in range(n)]
    P = [[float(i == j) for j in range(n)] for i in range(n)]
    sign = 1

    for k in range(n):
        # Выбор главного элемента
        p = max(range(k, n), key=lambda i: abs(U[i][k]))

        if abs(U[p][k]) < 1e-12:
            raise ValueError("Матрица вырождена")

        # Перестановка строк
        if p != k:
            U[k], U[p] = U[p], U[k]
            P[k], P[p] = P[p], P[k]

            # Переставляем ранее вычисленные элементы L
            for j in range(k):
                L[k][j], L[p][j] = L[p][j], L[k][j]

            sign *= -1

        # Исключение элементов под главным элементом
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]

            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    return L, U, P, sign


def solve_lu(L, U, P, b):
    n = len(b)

    # Перестановка правой части: Pb
    pb = [
        sum(P[i][j] * b[j] for j in range(n))
        for i in range(n)
    ]

    # Прямая подстановка: Ly = Pb
    y = [0.0] * n

    for i in range(n):
        y[i] = pb[i] - sum(
            L[i][j] * y[j] for j in range(i)
        )

    # Обратная подстановка: Ux = y
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        x[i] = (
            y[i] - sum(U[i][j] * x[j]
                       for j in range(i + 1, n))
        ) / U[i][i]

    return x


# LU-разложение
L, U, P, sign = lu_decomposition(A)

# Решение СЛАУ
x = solve_lu(L, U, P, b)

# Определитель: det(A) = det(P) * det(L) * det(U)
# det(L) = 1, det(P) = sign
det_A = sign
for i in range(len(A)):
    det_A *= U[i][i]

# Обратная матрица: решаем A*x_j = e_j
n = len(A)
A_inv = [[0.0] * n for _ in range(n)]

for j in range(n):
    e = [0.0] * n
    e[j] = 1.0
    column = solve_lu(L, U, P, e)

    for i in range(n):
        A_inv[i][j] = column[i]

# Вывод результатов
np.set_printoptions(precision=6, suppress=True)

print("Матрица L:")
print(np.array(L))

print("\nМатрица U:")
print(np.array(U))

print("\nМатрица перестановок P:")
print(np.array(P))

print("\nРешение СЛАУ:")
print(np.array(x))

print("\nОпределитель det(A):")
print(det_A)

print("\nОбратная матрица A^(-1):")
print(np.array(A_inv))

print("\nПроверка A*x = b:")
print(A @ np.array(x))
