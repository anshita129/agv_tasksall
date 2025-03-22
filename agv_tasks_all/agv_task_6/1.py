import numpy as np
import matplotlib.pyplot as plt

points = np.load('some_corresp.npz')
keypoints1 = points["pts1"]
keypoints2 = points["pts2"]
image1 = plt.imread('im1.png')
image2 = plt.imread('im2.png')

def refine_fundamental_matrix(fundamental_matrix):
    U, singular_values, Vt = np.linalg.svd(fundamental_matrix)
    singular_values[-1] = 0
    return U @ np.diag(singular_values) @ Vt

def compute_fundamental_matrix(points1, points2, scale):
    transform_matrix = np.array([[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1]])
    points1_normalized = np.column_stack((points1, np.ones(len(points1)))) @ transform_matrix
    points2_normalized = np.column_stack((points2, np.ones(len(points2)))) @ transform_matrix

    x1, y1 = points1_normalized[:, 0], points1_normalized[:, 1]
    x2, y2 = points2_normalized[:, 0], points2_normalized[:, 1]
    matrix_A = np.column_stack((x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, np.ones_like(x1)))

    _, _, Vt = np.linalg.svd(matrix_A)
    fundamental_matrix = Vt[-1].reshape(3, 3)

    fundamental_matrix = refine_fundamental_matrix(fundamental_matrix)
    fundamental_matrix = transform_matrix.T @ fundamental_matrix @ transform_matrix

    return fundamental_matrix

camera_intrinsics = np.load('intrinsics.npz')
max_dimension = max(image1.shape)
fundamental_matrix = compute_fundamental_matrix(keypoints1, keypoints2, max_dimension)

K1 = camera_intrinsics['K1']
K2 = camera_intrinsics['K2']
essential_matrix = K2.T @ fundamental_matrix @ K1
print(essential_matrix)
