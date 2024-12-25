from matplotlib import pyplot as plt

from smrs import smrs
from plot_svm import plot_svm
import copy
import numpy as np
import cvxpy as cp
from scipy.io import loadmat

'''
This file is for experimenting synthetic experiment using convexhull dataset used in HW3.
1. Load both separable and overlapping dataset
2. For separable, preprocess the dataset using only representative data
3. For overlap, preprocess the dataset by removing the representative data
4. Perform classification for two preprocessed data using SVM and plot.
'''


#load data from synthetic data
def load_data(train_path, test_path):
    train_sep = loadmat(train_path)
    test_sep = loadmat(test_path)
    return train_sep, test_sep

def extract_representatives(data, alpha):
    repInd, C = smrs(data, alpha, sparsity = 0.2)
    return repInd

def remove_representatives(data, rep_indices):
    all_indices = np.arange(data.shape[1])
    ex_indices = np.setdiff1d(all_indices, rep_indices)
    return data[:, ex_indices]

def solve_optimization(A, B):
    m = A.shape[1]
    n = B.shape[1]

    u = cp.Variable(m)
    v = cp.Variable(n)

    objective = cp.Minimize(0.5 * cp.norm(A @ u - B @ v, 2) ** 2)
    constraints = [
        cp.sum(u) == 1,
        cp.sum(v) == 1,
        u >= 0,
        v >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    return u.value, v.value

def overlapping(train_sep, test_sep, alpha=5):

    # Extract representative frames
    Y_A = train_sep['A']
    Y_B = train_sep['B']
    repInd_A = extract_representatives(Y_A, alpha)
    repInd_B = extract_representatives(Y_B, alpha)

    # Remove representatives
    #Y_A = remove_representatives(Y_A,repInd_A)
    #Y_B = remove_representatives(Y_B,repInd_B)
    Y_A = Y_A[:, repInd_A]
    Y_B = Y_B[:, repInd_B]

    # Solve optimization problem
    u_opt, v_opt = solve_optimization(Y_A, Y_B)
    return Y_A, Y_B, u_opt, v_opt

def separable(train_sep, test_sep, alpha=20):

    # Extract representative frames
    Y_A = train_sep['A']
    Y_B = train_sep['B']

    repInd_A = extract_representatives(Y_A, alpha)
    repInd_B = extract_representatives(Y_B, alpha)

    Y_A = Y_A[:, repInd_A]
    Y_B = Y_B[:, repInd_B]

    train_sep['A'] = Y_A
    train_sep['B'] = Y_B

    # Solve optimization problem
    u_opt, v_opt = solve_optimization(Y_A, Y_B)
    return Y_A, Y_B, u_opt, v_opt


def rm_rep(data, sInd):
    thr = 0.95
    Ys = data[:, sInd]

    # Number of selected columns
    Ns = Ys.shape[1]

    # Initialize the distance matrix
    d = np.zeros([Ns, Ns])

    # Compute the distance matrix
    for i in range(Ns - 1):
        for j in range(i + 1, Ns):
            d[i, j] = np.linalg.norm(Ys[:, i] - Ys[:, j])

    # Make the matrix symmetric
    d = d + d.T
    # Sort distances in descending order and get sorted indices
    dsorti = np.argsort(d, axis=0)[::-1]
    dsort = np.flipud(np.sort(d, axis=0))

    # Initialize array of indices
    pind = np.arange(0, Ns)

    # Remove redundant columns
    for i in range(Ns):
        if np.any(pind == i):
            cum = 0
            t = -1
            while cum <= (thr * np.sum(dsort[:, i])):
                t += 1
                cum += dsort[t, i]
            to_remove = np.setdiff1d(dsorti[t:, i], np.arange(0, i + 1), assume_unique=True)
            pind = np.setdiff1d(pind, to_remove, assume_unique=True)

    # Map remaining indices back to the original indices
    ind = sInd[pind]
    return ind

if __name__ == '__main__':
    train_path_sep = 'separable_case/train_separable.mat'
    test_path_sep = 'separable_case/test_separable.mat'

    train_path_ov  = 'overlap_case/train_overlap.mat'
    test_path_ov = 'overlap_case/test_overlap.mat'

    # Load data
    train_sep, test_sep = load_data(train_path_sep, test_path_sep)
    train_ov, test_ov = load_data(train_path_ov,test_path_ov)
    #leave the origin for comparison
    train_sep_origin = copy.deepcopy(train_sep)
    train_ov_origin = copy.deepcopy(train_ov)

    A_sep, B_sep, u_opt_sep, v_opt_sep = separable(train_sep, test_sep)
    A_ov, B_ov, u_opt_ov, v_opt_ov = overlapping(train_ov, test_ov)

    print(u_opt_ov)
    print(v_opt_ov)
    # Plot results
    train_data_sep = {'A': A_sep, 'B': B_sep}
    train_data_ov = {'A': A_ov, 'B': B_ov}

    u_origin_sep,v_origin_sep = solve_optimization(train_sep_origin['A'],train_sep_origin['B'])
    u_origin_ov,v_origin_ov = solve_optimization(train_ov_origin['A'],train_ov_origin['B'])

    plot_svm(u_opt_sep, v_opt_sep, train_data_sep, test_sep)
    plot_svm(u_opt_ov, v_opt_ov, train_data_ov, test_ov)
    plot_svm(u_origin_sep, v_origin_sep, train_sep_origin, test_sep)
    plot_svm(u_origin_ov, v_origin_ov, train_ov_origin, test_ov)
