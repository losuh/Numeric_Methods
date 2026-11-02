# Исходные данные
import numpy as np

A = np.array([
    [-5, -1, -3, -1],
    [-2,  0,  8, -4],
    [-7, -2,  2, -2],
    [ 2, -4, -4,  4]
], dtype=float)
b = np.array([18, -12, 6, -12], dtype=float)
# A = np.array([
#     [10, 1, 1],
#     [2, 10, 1],
#     [2, 2, 10]
# ])
# b = np.array([12, 13, 14], dtype=float)
l = len(A)

# LU-разложение
A_prev = A.astype(float).copy()
L = np.eye(l)
for k in range(l - 1):
    A_next = A_prev.copy()

    for i in range(k + 1, l):
        mu_i = A_prev[i, k] / A_prev[k, k]
        L[i, k] = mu_i
        for j in range(k, l):
            A_next[i, j] = A_prev[i, j] - mu_i * A_prev[k, j]
        A_next[i, k] = 0.0

    A_prev = A_next

U = A_prev.copy()
print("Нижняя треугольная матрица L:\n", L)
print("Верхняя треугольная матрица U:\n", U)


# Прямой ход
z = np.array([0] * l, dtype=float)
z[0] = b[0]
for i in range(1, l):
    s = 0
    for j in range(i):
        s += L[i, j] * z[j]
    z[i] = b[i] - s

# Обратный ход
x = np.array([0] * l, dtype=float)
x[l - 1] = z[l - 1] / U[l - 1][l - 1]
for i in range(l - 1, -1, -1):
    s = 0
    for j in range(i + 1, l):
        s += U[i, j] * x[j]
    x[i] = (z[i] - s) / U[i][i]
print("Решение системы (компоненты x по порядку):", x)
