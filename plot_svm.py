import numpy as np
import matplotlib.pyplot as plt
'''
This function plots svm for synthetic experiment.
'''
def plot_svm(u, v, train_data, test_data):
    A = train_data['A']
    B = train_data['B']
    X_test = test_data['X_test']
    true_labels = test_data['true_labels']


    closest_point_A = A @ u
    closest_point_B = B @ v

    print(f"Au: [{closest_point_A[0]:.6f}, {closest_point_A[1]:.6f}]")
    print(f"Bv: [{closest_point_B[0]:.6f}, {closest_point_B[1]:.6f}]")
    objective_value = 0.5 * np.linalg.norm(closest_point_A - closest_point_B)**2
    print(f"Objective value: {objective_value:.6f}")

    normal_vector = closest_point_A - closest_point_B
    classifier_boundary = (closest_point_A + closest_point_B) / 2

    true_labels = true_labels.flatten()

    plt.figure(figsize=(8, 6))
    plt.scatter(A[0, :], A[1, :], color='red', label='Class A', alpha=0.7)
    plt.scatter(B[0, :], B[1, :], color='blue', label='Class B', alpha=0.7)
    plt.scatter(closest_point_A[0], closest_point_A[1], color='red', edgecolor='black', label='Au', s=100, linewidth=2)
    plt.scatter(closest_point_B[0], closest_point_B[1], color='blue', edgecolor='black', label='Bv', s=100, linewidth=2)

    plt.plot([closest_point_A[0], closest_point_B[0]],
             [closest_point_A[1], closest_point_B[1]], 'k-', linewidth=1.5, label='Line Segment')

    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    slope = -normal_vector[0] / normal_vector[1]
    intercept = classifier_boundary[1] - slope * classifier_boundary[0]

    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = slope * x_vals + intercept

    plt.plot(x_vals, y_vals, 'k--', linewidth=2, label='Classifier Boundary')

    plt.title('Training Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[0, true_labels == 1], X_test[1, true_labels == 1], color='red', label='Class A', alpha=0.7)

    plt.scatter(X_test[0, true_labels == -1], X_test[1, true_labels == -1], color='blue', label='Class B', alpha=0.7)

    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = slope * x_vals + intercept

    plt.plot(x_vals, y_vals, 'k--', linewidth=3, label='Classifier Boundary')

    plt.title('Testing Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    predicted_labels = np.sign((X_test.T - classifier_boundary ) @ normal_vector)

    classification_error = np.sum(predicted_labels != true_labels) / len(true_labels)
    print(f"Classification Error on Testing Data: {classification_error * 100:.3f}%")
