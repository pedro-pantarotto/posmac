# Método de Jacobi-Richardson

def jacobi(A, b, x0, tol=1e-10, max_iter=100):
    """
    Método de Jacobi-Richardson para resolver sistemas lineares Ax = b.
    A: matriz dos coeficientes
    b: vetor constante
    x0: chute inicial
    tol: tolerância para o critério de parada
    max_iter: número máximo de iterações
    """
    n = len(A)
    x = x0.copy()
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        # Verifica convergência
        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            break
        x = x_new
    return x


# Método dos Gradientes

def metodo_dos_gradientes(A, b, x0, tol=1e-10, max_iter=100):
    """
    Método dos Gradientes para resolver Ax = b.
    A: matriz simétrica definida positiva
    b: vetor constante
    x0: chute inicial
    """
    import numpy as np
    x = x0
    r = b - A @ x
    for k in range(max_iter):
        alpha = (r @ r) / (r @ (A @ r))
        x_new = x + alpha * r
        r_new = r - alpha * (A @ r)
        if np.linalg.norm(r_new) < tol:
            break
        x = x_new
        r = r_new
    return x


# Método dos Gradientes Conjugados

def gradientes_conjugados(A, b, x0, tol=1e-10, max_iter=100):
    """
    Método dos Gradientes Conjugados para Ax = b.
    A: matriz simétrica definida positiva
    b: vetor constante
    x0: chute inicial
    """
    import numpy as np
    x = x0
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x
