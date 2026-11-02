# Исходные данные
import numpy as np

# Матрица коэффициентов A
A = np.array([
    [18,  -9,   0,   0,   0],
    [ 2,  -9,  -4,   0,   0],
    [ 0,  -9,  21,  -8,   0],
    [ 0,   0,  -4, -10,   5],
    [ 0,   0,   0,   7,  12]
], dtype=float)

# Вектор правых частей
B = np.array([-81, 71, -39, 64, 3], dtype=float)

n = len(B)

# Диагонали матрицы
a = [A[i, i - 1] if i > 0 else 0 for i in range(n)]
b = [A[i, i] for i in range(n)]
c = [A[i, i + 1] if i < n - 1 else 0 for i in range(n)]

# Начальные коэффициенты прогонки
P = np.array([0] * n, dtype=float)
Q = np.array([0] * n, dtype=float)
for i in range(n):
    P[i] = -c[i] / b[i]
    Q[i] = B[i] / b[i]

# Прямой ход
P[0] = -c[0] / b[0]
Q[0] = B[0] / b[0]

for i in range(1, n - 1):
    P[i] = -c[i] / (b[i] + a[i] * P[i - 1])
    Q[i] = (B[i] - a[i] * Q[i - 1]) / (b[i] + a[i] * P[i - 1])

P[n - 1] = 0
Q[n - 1] = (B[n - 1] - a[n - 1] * Q[n - 2]) / (b[n - 1] + a[n - 1] * P[n - 2])


# Обратный ход
x = np.array([0] * n, dtype=float)
x[n - 1] = Q[n - 1]
for i in range(n - 2, -1, -1):
    x[i] = P[i] * x[i + 1] + Q[i]
print("Решение системы:", x)
