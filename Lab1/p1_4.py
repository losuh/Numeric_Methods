import numpy as np

# Исходные данные. Вариант 16
A = np.array([
    [8, -3, 9],
    [-3, 8, -2],
    [9, -2, -8]
], dtype=float)
eps = 1e-6
n = len(A)

# Начальная матрица и накопленное произведение матриц вращения
A_prev = A.copy()
U = np.eye(n)
iterations = 0

# Критерий остановки: корень из суммы квадратов элементов выше диагонали
s = 0
for i in range(n):
    for j in range(i + 1, n):
        s += A_prev[i, j] ** 2
error = np.sqrt(s)

print("Итерация    t(A) — оценка погрешности")
print(f"{iterations:<11} {error:.10e}")
while error > eps:
    # Максимальный по модулю элемент выше главной диагонали
    i_max, j_max = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A_prev[i, j]) > abs(A_prev[i_max, j_max]):
                i_max, j_max = i, j

    # Угол вращения по формуле из методички
    if A_prev[i_max, i_max] == A_prev[j_max, j_max]:
        phi = np.pi / 4
    else:
        phi = 0.5 * np.arctan(
            2 * A_prev[i_max, j_max]
            / (A_prev[i_max, i_max] - A_prev[j_max, j_max])
        )

    # Матрица вращения: U_ij = -sin(phi), U_ji = sin(phi)
    U_k = np.eye(n)
    U_k[i_max, i_max] = np.cos(phi)
    U_k[j_max, j_max] = np.cos(phi)
    U_k[i_max, j_max] = -np.sin(phi)
    U_k[j_max, i_max] = np.sin(phi)

    # Преобразование подобия и накопление собственных векторов
    A_next = U_k.T @ A_prev @ U_k
    U = U @ U_k
    A_prev = A_next
    iterations += 1

    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += A_prev[i, j] ** 2
    error = np.sqrt(s)
    print(f"{iterations:<11} {error:.10e}")

# Собственные значения — диагональ, собственные векторы — столбцы U
eigenvalues = np.diag(A_prev).copy()
print("\nСобственные значения:")
print(eigenvalues)
print("\nСобственные векторы (столбцы):")
print(U)
print("\nКоличество итераций:", iterations)

