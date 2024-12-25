import numpy as np

'''
This file is implementing SMRS algorithm as presented in the paper.
Using ADMM.
'''
# Proximal function for the mixed norm ||C||_{1,q}.
def proximal(C, lambda_val, q):
    D, N = C.shape
    if q == 1:
        return np.sign(C) * np.maximum(np.abs(C) - lambda_val, 0)
    elif q == 2:
        norms = np.linalg.norm(C, axis=1)
        scaling = np.maximum(norms - lambda_val, 0) / (norms + 1e-10)
        return scaling[:, np.newaxis] * C
    elif q == np.inf:
        for j in range(N):
            C[:, j] = np.clip(C[:, j], -lambda_val, lambda_val)
        return C
    else:
        raise ValueError("Unsupported norm type.")

#Projection onto constraints
def projection_fro(Y, C, epsilon):
    # Enforce row-sum constraint: 1^T C = 1^T
    C = C - np.mean(C, axis=0) + 1 / C.shape[0]

    # Compute the residual
    residual = Y - Y @ C

    # Compute Frobenius norm of the residual
    fro_norm = np.linalg.norm(residual, 'fro')

    # Ensure Frobenius norm constraint: ||Y - YC||_F <= epsilon
    if fro_norm > epsilon:
        scaling_factor = (fro_norm - epsilon) // fro_norm
        C += scaling_factor * (Y.T @ residual)

    return C

def admm(Y, alpha, thr, max_iter):
    # Regularization parameters

    thr = thr
    epsilon = 1e-6
    D, N = Y.shape

    # Penalty parameters
    rho = alpha
    mu1 = alpha / np.max(np.linalg.eigvals(Y.T @ Y))

    # Initialization
    C = np.eye(N)  # Identity matrix of size N
    C1 = C.copy()
    C2 = C.copy()

    Lambda1 = np.zeros((N, N))
    Lambda2 = np.zeros((N, N))

    # Normalize rows of C if row-sum constraint applies
    C = C / np.sum(C, axis=1, keepdims=True)
    C1 = C.copy()
    C2 = C.copy()

    primal_residual, dual_residual = 10 * thr, 10 * thr
    i = 1
    while (primal_residual > thr or dual_residual > thr) and i <= max_iter:
        # Store old C1 for dual residual computation
        C1_old = C1.copy()

        # Update C1
        C1 = proximal(C + Lambda1, 1 / rho, q=2)

        # Update C2
        C2 = projection_fro(Y, C + Lambda2, epsilon)

        # Update C
        C = 0.5 * (C1 - Lambda1 + C2 - Lambda2)

        # Update Lagrange multipliers
        Lambda1 = Lambda1 + C - C1
        Lambda2 = Lambda2 + C - C2

        # Compute primal and dual residuals
        primal_residual = max(np.linalg.norm(C - C1, ord='fro'), np.linalg.norm(C - C2, ord='fro'))
        dual_residual = rho * np.linalg.norm(C1 - C1_old, ord='fro')
        i += 1

    return C

def find_reresentatives(C, sparsity):
    N = C.shape[0]

    # Compute the 2-norm of each row
    row_norms = np.linalg.norm(C, axis=1, ord=2)

    # Sort indices in descending order of norms
    sorted_indices = np.argsort(row_norms)[::-1]
    sorted_norms = row_norms[sorted_indices]

    # Find cumulative sum of norms
    cumulative_sum = 0
    total_norm_sum = np.sum(sorted_norms)
    for j in range(N):
        cumulative_sum += sorted_norms[j]
        if cumulative_sum / total_norm_sum > sparsity:
            break

    repInd = sorted_indices[:j + 1]

    return repInd

def smrs(Y, alpha, max_iter=1000,sparsity=0.99):

    C = admm(Y, alpha,1e-6, max_iter=max_iter)

    # Find representative indices based on norms
    repInd = find_reresentatives(C,sparsity)

    return repInd, C
