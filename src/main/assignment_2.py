import numpy as np

def nevilles_method(x_points, y_points, x):
    n = len(x_points)
    Q = np.zeros((n, n))
    Q[:, 0] = y_points
    
    for i in range(1, n):
        for j in range(n - i):
            Q[j, i] = ((x - x_points[j + i]) * Q[j, i - 1] - (x - x_points[j]) * Q[j + 1, i - 1]) / (x_points[j] - x_points[j + i])
    
    return Q[0, -1]

def newtons_forward_difference(x_points, y_points):
    n = len(x_points)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_points
    
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = (diff_table[i + 1, j - 1] - diff_table[i, j - 1]) / (x_points[i + j] - x_points[i])
    
    return diff_table[0, :]

def newtons_interpolation(x_points, y_points, x):
    coeffs = newtons_forward_difference(x_points, y_points)
    n = len(coeffs)
    result = coeffs[0]
    product_term = 1
    
    for i in range(1, n):
        product_term *= (x - x_points[i - 1])
        result += coeffs[i] * product_term
    
    return result

def hermite_interpolation(x_points, y_points, derivatives):
    n = len(x_points)
    m = 2 * n
    Q = np.zeros((m, m))
    z = np.zeros(m)
    
    for i in range(n):
        z[2*i] = z[2*i + 1] = x_points[i]
        Q[2*i, 0] = Q[2*i + 1, 0] = y_points[i]
        Q[2*i + 1, 1] = derivatives[i]
        if i != 0:
            Q[2*i, 1] = (Q[2*i, 0] - Q[2*i - 1, 0]) / (z[2*i] - z[2*i - 1])
    
    for j in range(2, m):
        for i in range(m - j):
            Q[i, j] = (Q[i + 1, j - 1] - Q[i, j - 1]) / (z[i + j] - z[i])
    
    return Q

def cubic_spline_matrix(x_points, y_points):
    n = len(x_points) - 1
    h = np.diff(x_points)
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    
    A[0, 0] = A[n, n] = 1
    
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = (3 / h[i]) * (y_points[i + 1] - y_points[i]) - (3 / h[i - 1]) * (y_points[i] - y_points[i - 1])
    
    return A, b, np.linalg.solve(A, b)

def main():
    # Question 1
    x_q1 = np.array([3.6, 3.8, 3.9])
    y_q1 = np.array([1.675, 1.436, 1.318])
    print("Neville's Method:", nevilles_method(x_q1, y_q1, 3.7))
    
    # Question 2
    x_q2 = np.array([7.2, 7.4, 7.5, 7.6])
    y_q2 = np.array([23.5492, 25.3913, 26.8224, 27.4589])
    print("Newton's Forward Differences:", newtons_forward_difference(x_q2, y_q2))
    
    # Question 3
    print("Newton's Interpolation at x=7.3:", newtons_interpolation(x_q2, y_q2, 7.3))
    
    # Question 4
    x_q4 = np.array([3.6, 3.8, 3.9])
    y_q4 = np.array([1.675, 1.436, 1.318])
    dy_q4 = np.array([-1.195, -1.188, -1.182])
    print("Hermite Interpolation Table:", hermite_interpolation(x_q4, y_q4, dy_q4))
    
    # Question 5
    x_q5 = np.array([2, 5, 8, 10])
    y_q5 = np.array([3, 5, 7, 9])
    A, b, x = cubic_spline_matrix(x_q5, y_q5)
    print("Cubic Spline Matrix A:", A)
    print("Vector b:", b)
    print("Vector x:", x)
    
if __name__ == "__main__":
    main()

