import numpy as np

def is_positive_definite(A):
    eigenvalues = np.linalg.eigvals(A)
    print("Eigenvalues:", eigenvalues)
    return np.all(eigenvalues > 0)


def hessian_func_example(x, y):
    return np.array([[12 * x**2, -1],
                     [-1, 2]])

def hessian_eigenvalues_at_point(Hessian_func, point):
    H = Hessian_func(*point)
    eigenvalues = np.linalg.eigvals(H)
    print(f"Hessian at {point}:\n{H}")
    print(f"Eigenvalues: {eigenvalues}")
    return eigenvalues


def hessian_f1(x, y):
    # f(x, y) = x^3 + 2y^3 - x*y
    # ∂²f/∂x² = 6x, ∂²f/∂y² = 12y, ∂²f/∂x∂y = ∂²f/∂y∂x = -1
    return np.array([[6 * x, -1],
                     [-1, 12 * y]])

def check_concavity(x, y):
    H = hessian_f1(x, y)
    eigvals = np.linalg.eigvals(H)
    print(f"Point: ({x}, {y})")
    print("Hessian:\n", H)
    print("Eigenvalues:", eigvals)
    if np.all(eigvals > 0):
        print("=> Local minimum (convex)")
    elif np.all(eigvals < 0):
        print("=> Local maximum (concave)")
    elif np.any(eigvals > 0) and np.any(eigvals < 0):
        print("=> Saddle point (neither)")
    else:
        print("=> Inconclusive (eigenvalue is zero)")


def gradient_f2(x, y):
    # f(x, y) = 4x + 2y - x^2 - 3y^2
    df_dx = 4 - 2 * x
    df_dy = 2 - 6 * y
    return np.array([df_dx, df_dy])

def hessian_f2(x, y):
    # Second partials: ∂²f/∂x² = -2, ∂²f/∂y² = -6, mixed = 0
    return np.array([[-2, 0],
                     [0, -6]])


def find_critical_points():
    # Solve gradient = 0
    # 4 - 2x = 0 → x = 2
    # 2 - 6y = 0 → y = 1/3
    x_crit, y_crit = 2, 1/3
    print(f"Critical point: ({x_crit}, {y_crit})")
    H = hessian_f2(x_crit, y_crit)
    eigvals = np.linalg.eigvals(H)
    print("Hessian at critical point:\n", H)
    print("Eigenvalues:", eigvals)
    if np.all(eigvals < 0):
        print("=> Local maximum")
    elif np.all(eigvals > 0):
        print("=> Local minimum")
    else:
        print("=> Saddle point or inconclusive")


# 1. Check positive definiteness
A = np.array([[9, -15], [-15, 21]])
print("Matrix A is positive definite:", is_positive_definite(A))

# 2. Hessian eigenvalues for given function
print("\nEigenvalues of Hessian at (3,1):")
hessian_eigenvalues_at_point(hessian_func_example, (3, 1))

# 3. Concavity of f(x, y) = x^3 + 2y^3 - xy at various points
print("\nConcavity at (0,0):")
check_concavity(0, 0)
print("\nConcavity at (3,3):")
check_concavity(3, 3)
print("\nConcavity at (3,-3):")
check_concavity(3, -3)

# 4. Gradient, critical points, and classification
print("\nAnalyzing f(x, y) = 4x + 2y - x^2 - 3y^2")
find_critical_points()
