import numpy as np

# Исходные данные
A = np.array([
    [21, -6, -9, -4],
    [-6, 20, -4, 2],
    [-2, -7, -20, 3],
    [4, 9, 6, 24]
], dtype=float)
b = np.array([127, -144, 236, -5], dtype=float)
eps = 1e-6
n = len(b)

# Приведение к виду x = beta + alpha * x (формула 1.18)
alpha = np.zeros((n, n))
beta = np.zeros(n)
for i in range(n):
    beta[i] = b[i] / A[i, i]
    for j in range(n):
        if i != j:
            alpha[i, j] = -A[i, j] / A[i, i]

print('alpha:', alpha)
print('beta:', beta)

# Норма матрицы — максимальная сумма модулей элементов строки.
# Для вектора используется согласованная норма max |x_i|.
alpha_norm = np.max(np.sum(np.abs(alpha), axis=1))
print("Норма alpha:", alpha_norm)

# Верхняя треугольная часть alpha для оценки погрешности метода Зейделя
C = np.triu(alpha)
C_norm = np.max(np.sum(np.abs(C), axis=1))

print("Норма C:", C_norm)

# Сравнение числа итераций при разной точности
print("\nТочность, Простые итерации, Метод Зейделя")
for tolerance in [1e-2, 1e-4, eps]:
    # Метод простых итераций, начальное приближение x^(0) = beta
    x = beta.copy()
    error = np.inf
    iterations = 0
    while error > tolerance:
        x_prev = x.copy()
        x = beta + alpha @ x_prev
        iterations += 1
        difference = np.max(np.abs(x - x_prev))
        if alpha_norm < 1:
            error = alpha_norm / (1 - alpha_norm) * difference
        else:
            error = difference
    x_iterations = x.copy()
    error_iterations = error

    # Метод Зейделя: уже найденные компоненты берутся с текущей итерации
    x = beta.copy()
    error = np.inf
    seidel_iterations = 0
    while error > tolerance:
        x_prev = x.copy()
        for i in range(n):
            s = 0
            for j in range(i):
                s += alpha[i, j] * x[j]
            for j in range(i, n):
                s += alpha[i, j] * x_prev[j]
            x[i] = beta[i] + s
        seidel_iterations += 1
        difference = np.max(np.abs(x - x_prev))
        if alpha_norm < 1:
            error = C_norm / (1 - alpha_norm) * difference
        else:
            error = difference
    x_seidel = x.copy()
    print(tolerance, iterations, seidel_iterations)

print("\nРешение методом простых итераций:")
print(x_iterations)
print("Оценка погрешности:", error_iterations)

print("\nРешение методом Зейделя:")
print(x_seidel)
print("Оценка погрешности:", error)
