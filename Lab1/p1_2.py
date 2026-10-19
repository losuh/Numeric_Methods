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

# Извлечение трех диагоналей матрицы
a = [A[i, i - 1] if i > 0 else 0 for i in range(n)]
b = [A[i, i] for i in range(n)]
c = [A[i, i + 1] if i < n - 1 else 0 for i in range(n)]

# Массивы коэффициентов прогонки
P = [0.0] * n
Q = [0.0] * n

# Прямой ход
P[0] = -c[0] / b[0]
Q[0] = B[0] / b[0]

for i in range(1, n):
    denominator = b[i] + a[i] * P[i - 1]

    P[i] = -c[i] / denominator
    Q[i] = (B[i] - a[i] * Q[i - 1]) / denominator

# Обратный ход
x = np.zeros(n)
x[n - 1] = Q[n - 1]

for i in range(n - 2, -1, -1):
    x[i] = P[i] * x[i + 1] + Q[i]

# Вывод результатов
print("Матрица A:")
print(A)

print("\nВектор B:")
print(B)

print("\nКоэффициенты P:")
for i in range(n):
    print(f"P[{i + 1}] = {P[i]:.6f}")

print("\nКоэффициенты Q:")
for i in range(n):
    print(f"Q[{i + 1}] = {Q[i]:.6f}")

print("\nРешение СЛАУ:")
for i in range(n):
    print(f"x[{i + 1}] = {x[i]:.6f}")

# Проверка решения
print("\nПроверка A @ x:")
print(A @ x)

print("\nВектор B:")
print(B)

print("\nПроверка:", np.allclose(A @ x, B))