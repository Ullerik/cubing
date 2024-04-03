import numpy as np
from numba import njit
import time
import matplotlib.pyplot as plt

# defined moves
move_dict = {
    # outer moves
    "R": np.array([0, 1, 2, 83, 4, 5, 6, 87, 8, 9, 10, 91, 12, 13, 14, 95, 28, 24, 20, 16, 29, 25, 21, 17, 30, 26, 22, 18, 31, 27, 23, 19, 79, 33, 34, 35, 75, 37, 38, 39, 71, 41, 42, 43, 67, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 3, 68, 69, 70, 7, 72, 73, 74, 11, 76, 77, 78, 15, 80, 81, 82, 44, 84, 85, 86, 40, 88, 89, 90, 36, 92, 93, 94, 32]),
    "R'": np.array([0, 1, 2, 67, 4, 5, 6, 71, 8, 9, 10, 75, 12, 13, 14, 79, 19, 23, 27, 31, 18, 22, 26, 30, 17, 21, 25, 29, 16, 20, 24, 28, 95, 33, 34, 35, 91, 37, 38, 39, 87, 41, 42, 43, 83, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 44, 68, 69, 70, 40, 72, 73, 74, 36, 76, 77, 78, 32, 80, 81, 82, 3, 84, 85, 86, 7, 88, 89, 90, 11, 92, 93, 94, 15]),
    "R2": np.array([0, 1, 2, 44, 4, 5, 6, 40, 8, 9, 10, 36, 12, 13, 14, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 33, 34, 35, 11, 37, 38, 39, 7, 41, 42, 43, 3, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 83, 68, 69, 70, 87, 72, 73, 74, 91, 76, 77, 78, 95, 80, 81, 82, 67, 84, 85, 86, 71, 88, 89, 90, 75, 92, 93, 94, 79]),
    "L": np.array([64, 1, 2, 3, 68, 5, 6, 7, 72, 9, 10, 11, 76, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 92, 36, 37, 38, 88, 40, 41, 42, 84, 44, 45, 46, 80, 60, 56, 52, 48, 61, 57, 53, 49, 62, 58, 54, 50, 63, 59, 55, 51, 47, 65, 66, 67, 43, 69, 70, 71, 39, 73, 74, 75, 35, 77, 78, 79, 0, 81, 82, 83, 4, 85, 86, 87, 8, 89, 90, 91, 12, 93, 94, 95]),
    "L'": np.array([80, 1, 2, 3, 84, 5, 6, 7, 88, 9, 10, 11, 92, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 76, 36, 37, 38, 72, 40, 41, 42, 68, 44, 45, 46, 64, 51, 55, 59, 63, 50, 54, 58, 62, 49, 53, 57, 61, 48, 52, 56, 60, 0, 65, 66, 67, 4, 69, 70, 71, 8, 73, 74, 75, 12, 77, 78, 79, 47, 81, 82, 83, 43, 85, 86, 87, 39, 89, 90, 91, 35, 93, 94, 95]),
    "L2": np.array([47, 1, 2, 3, 43, 5, 6, 7, 39, 9, 10, 11, 35, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 12, 36, 37, 38, 8, 40, 41, 42, 4, 44, 45, 46, 0, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 80, 65, 66, 67, 84, 69, 70, 71, 88, 73, 74, 75, 92, 77, 78, 79, 64, 81, 82, 83, 68, 85, 86, 87, 72, 89, 90, 91, 76, 93, 94, 95]),
    "U": np.array([16, 17, 18, 19, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 0, 1, 2, 3, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 76, 72, 68, 64, 77, 73, 69, 65, 78, 74, 70, 66, 79, 75, 71, 67, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "U'": np.array([48, 49, 50, 51, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 32, 33, 34, 35, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 67, 71, 75, 79, 66, 70, 74, 78, 65, 69, 73, 77, 64, 68, 72, 76, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "U2": np.array([32, 33, 34, 35, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 48, 49, 50, 51, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 16, 17, 18, 19, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "D": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 60, 61, 62, 63, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 44, 45, 46, 47, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 92, 88, 84, 80, 93, 89, 85, 81, 94, 90, 86, 82, 95, 91, 87, 83]),
    "D'": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 44, 45, 46, 47, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 60, 61, 62, 63, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 83, 87, 91, 95, 82, 86, 90, 94, 81, 85, 89, 93, 80, 84, 88, 92]),
    "D2": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 44, 45, 46, 47, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 60, 61, 62, 63, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 28, 29, 30, 31, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80]),
    "F": np.array([12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3, 76, 17, 18, 19, 77, 21, 22, 23, 78, 25, 26, 27, 79, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 80, 52, 53, 54, 81, 56, 57, 58, 82, 60, 61, 62, 83, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 63, 59, 55, 51, 28, 24, 20, 16, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "F'": np.array([3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12, 83, 17, 18, 19, 82, 21, 22, 23, 81, 25, 26, 27, 80, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 79, 52, 53, 54, 78, 56, 57, 58, 77, 60, 61, 62, 76, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 16, 20, 24, 28, 51, 55, 59, 63, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "F2": np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 63, 17, 18, 19, 59, 21, 22, 23, 55, 25, 26, 27, 51, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 28, 52, 53, 54, 24, 56, 57, 58, 20, 60, 61, 62, 16, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 83, 82, 81, 80, 79, 78, 77, 76, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "B": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 95, 20, 21, 22, 94, 24, 25, 26, 93, 28, 29, 30, 92, 44, 40, 36, 32, 45, 41, 37, 33, 46, 42, 38, 34, 47, 43, 39, 35, 67, 49, 50, 51, 66, 53, 54, 55, 65, 57, 58, 59, 64, 61, 62, 63, 19, 23, 27, 31, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 48, 52, 56, 60]),
    "B'": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 64, 20, 21, 22, 65, 24, 25, 26, 66, 28, 29, 30, 67, 35, 39, 43, 47, 34, 38, 42, 46, 33, 37, 41, 45, 32, 36, 40, 44, 92, 49, 50, 51, 93, 53, 54, 55, 94, 57, 58, 59, 95, 61, 62, 63, 60, 56, 52, 48, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 31, 27, 23, 19]),
    "B2": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 60, 20, 21, 22, 56, 24, 25, 26, 52, 28, 29, 30, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 49, 50, 51, 27, 53, 54, 55, 23, 57, 58, 59, 19, 61, 62, 63, 95, 94, 93, 92, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 67, 66, 65, 64]),
    
    # rotations
    "x": np.array([80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 28, 24, 20, 16, 29, 25, 21, 17, 30, 26, 22, 18, 31, 27, 23, 19, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 51, 55, 59, 63, 50, 54, 58, 62, 49, 53, 57, 61, 48, 52, 56, 60, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32]),
    "x'": np.array([64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 19, 23, 27, 31, 18, 22, 26, 30, 17, 21, 25, 29, 16, 20, 24, 28, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 60, 56, 52, 48, 61, 57, 53, 49, 62, 58, 54, 50, 63, 59, 55, 51, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    "x2": np.array([47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]),
    "y": np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 76, 72, 68, 64, 77, 73, 69, 65, 78, 74, 70, 66, 79, 75, 71, 67, 83, 87, 91, 95, 82, 86, 90, 94, 81, 85, 89, 93, 80, 84, 88, 92]),
    "y'": np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 67, 71, 75, 79, 66, 70, 74, 78, 65, 69, 73, 77, 64, 68, 72, 76, 92, 88, 84, 80, 93, 89, 85, 81, 94, 90, 86, 82, 95, 91, 87, 83]),
    "y2": np.array([32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80]),
    "z": np.array([12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3, 76, 72, 68, 64, 77, 73, 69, 65, 78, 74, 70, 66, 79, 75, 71, 67, 35, 39, 43, 47, 34, 38, 42, 46, 33, 37, 41, 45, 32, 36, 40, 44, 92, 88, 84, 80, 93, 89, 85, 81, 94, 90, 86, 82, 95, 91, 87, 83, 60, 56, 52, 48, 61, 57, 53, 49, 62, 58, 54, 50, 63, 59, 55, 51, 28, 24, 20, 16, 29, 25, 21, 17, 30, 26, 22, 18, 31, 27, 23, 19]),
    "z'": np.array([3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12, 83, 87, 91, 95, 82, 86, 90, 94, 81, 85, 89, 93, 80, 84, 88, 92, 44, 40, 36, 32, 45, 41, 37, 33, 46, 42, 38, 34, 47, 43, 39, 35, 67, 71, 75, 79, 66, 70, 74, 78, 65, 69, 73, 77, 64, 68, 72, 76, 19, 23, 27, 31, 18, 22, 26, 30, 17, 21, 25, 29, 16, 20, 24, 28, 51, 55, 59, 63, 50, 54, 58, 62, 49, 53, 57, 61, 48, 52, 56, 60]),
    "z2": np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64]),
    
    # wide moves
    "Rw": np.array([0, 1, 82, 83, 4, 5, 86, 87, 8, 9, 90, 91, 12, 13, 94, 95, 28, 24, 20, 16, 29, 25, 21, 17, 30, 26, 22, 18, 31, 27, 23, 19, 79, 78, 34, 35, 75, 74, 38, 39, 71, 70, 42, 43, 67, 66, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 2, 3, 68, 69, 6, 7, 72, 73, 10, 11, 76, 77, 14, 15, 80, 81, 45, 44, 84, 85, 41, 40, 88, 89, 37, 36, 92, 93, 33, 32]),
    "Rw'": np.array([0, 1, 66, 67, 4, 5, 70, 71, 8, 9, 74, 75, 12, 13, 78, 79, 19, 23, 27, 31, 18, 22, 26, 30, 17, 21, 25, 29, 16, 20, 24, 28, 95, 94, 34, 35, 91, 90, 38, 39, 87, 86, 42, 43, 83, 82, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 45, 44, 68, 69, 41, 40, 72, 73, 37, 36, 76, 77, 33, 32, 80, 81, 2, 3, 84, 85, 6, 7, 88, 89, 10, 11, 92, 93, 14, 15]),
    "Rw2": np.array([0, 1, 45, 44, 4, 5, 41, 40, 8, 9, 37, 36, 12, 13, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 34, 35, 11, 10, 38, 39, 7, 6, 42, 43, 3, 2, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 82, 83, 68, 69, 86, 87, 72, 73, 90, 91, 76, 77, 94, 95, 80, 81, 66, 67, 84, 85, 70, 71, 88, 89, 74, 75, 92, 93, 78, 79]),
    "Lw": np.array([64, 65, 2, 3, 68, 69, 6, 7, 72, 73, 10, 11, 76, 77, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 93, 92, 36, 37, 89, 88, 40, 41, 85, 84, 44, 45, 81, 80, 60, 56, 52, 48, 61, 57, 53, 49, 62, 58, 54, 50, 63, 59, 55, 51, 47, 46, 66, 67, 43, 42, 70, 71, 39, 38, 74, 75, 35, 34, 78, 79, 0, 1, 82, 83, 4, 5, 86, 87, 8, 9, 90, 91, 12, 13, 94, 95]),
    "Lw'": np.array([80, 81, 2, 3, 84, 85, 6, 7, 88, 89, 10, 11, 92, 93, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 77, 76, 36, 37, 73, 72, 40, 41, 69, 68, 44, 45, 65, 64, 51, 55, 59, 63, 50, 54, 58, 62, 49, 53, 57, 61, 48, 52, 56, 60, 0, 1, 66, 67, 4, 5, 70, 71, 8, 9, 74, 75, 12, 13, 78, 79, 47, 46, 82, 83, 43, 42, 86, 87, 39, 38, 90, 91, 35, 34, 94, 95]),
    "Lw2": np.array([47, 46, 2, 3, 43, 42, 6, 7, 39, 38, 10, 11, 35, 34, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 13, 12, 36, 37, 9, 8, 40, 41, 5, 4, 44, 45, 1, 0, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 80, 81, 66, 67, 84, 85, 70, 71, 88, 89, 74, 75, 92, 93, 78, 79, 64, 65, 82, 83, 68, 69, 86, 87, 72, 73, 90, 91, 76, 77, 94, 95]),
    "Uw": np.array([16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 40, 41, 42, 43, 44, 45, 46, 47, 0, 1, 2, 3, 4, 5, 6, 7, 56, 57, 58, 59, 60, 61, 62, 63, 76, 72, 68, 64, 77, 73, 69, 65, 78, 74, 70, 66, 79, 75, 71, 67, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "Uw'": np.array([48, 49, 50, 51, 52, 53, 54, 55, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 40, 41, 42, 43, 44, 45, 46, 47, 32, 33, 34, 35, 36, 37, 38, 39, 56, 57, 58, 59, 60, 61, 62, 63, 67, 71, 75, 79, 66, 70, 74, 78, 65, 69, 73, 77, 64, 68, 72, 76, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "Uw2": np.array([32, 33, 34, 35, 36, 37, 38, 39, 8, 9, 10, 11, 12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 40, 41, 42, 43, 44, 45, 46, 47, 16, 17, 18, 19, 20, 21, 22, 23, 56, 57, 58, 59, 60, 61, 62, 63, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "Dw": np.array([0, 1, 2, 3, 4, 5, 6, 7, 56, 57, 58, 59, 60, 61, 62, 63, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 40, 41, 42, 43, 44, 45, 46, 47, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 92, 88, 84, 80, 93, 89, 85, 81, 94, 90, 86, 82, 95, 91, 87, 83]),
    "Dw'": np.array([0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 40, 41, 42, 43, 44, 45, 46, 47, 32, 33, 34, 35, 36, 37, 38, 39, 56, 57, 58, 59, 60, 61, 62, 63, 48, 49, 50, 51, 52, 53, 54, 55, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 83, 87, 91, 95, 82, 86, 90, 94, 81, 85, 89, 93, 80, 84, 88, 92]),
    "Dw2": np.array([0, 1, 2, 3, 4, 5, 6, 7, 40, 41, 42, 43, 44, 45, 46, 47, 16, 17, 18, 19, 20, 21, 22, 23, 56, 57, 58, 59, 60, 61, 62, 63, 32, 33, 34, 35, 36, 37, 38, 39, 8, 9, 10, 11, 12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55, 24, 25, 26, 27, 28, 29, 30, 31, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80]),
    "Fw": np.array([12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3, 76, 72, 18, 19, 77, 73, 22, 23, 78, 74, 26, 27, 79, 75, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 84, 80, 52, 53, 85, 81, 56, 57, 86, 82, 60, 61, 87, 83, 64, 65, 66, 67, 68, 69, 70, 71, 62, 58, 54, 50, 63, 59, 55, 51, 28, 24, 20, 16, 29, 25, 21, 17, 88, 89, 90, 91, 92, 93, 94, 95]),
    "Fw'": np.array([3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12, 83, 87, 18, 19, 82, 86, 22, 23, 81, 85, 26, 27, 80, 84, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 75, 79, 52, 53, 74, 78, 56, 57, 73, 77, 60, 61, 72, 76, 64, 65, 66, 67, 68, 69, 70, 71, 17, 21, 25, 29, 16, 20, 24, 28, 51, 55, 59, 63, 50, 54, 58, 62, 88, 89, 90, 91, 92, 93, 94, 95]),
    "Fw2": np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 63, 62, 18, 19, 59, 58, 22, 23, 55, 54, 26, 27, 51, 50, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 29, 28, 52, 53, 25, 24, 56, 57, 21, 20, 60, 61, 17, 16, 64, 65, 66, 67, 68, 69, 70, 71, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 88, 89, 90, 91, 92, 93, 94, 95]),
    "Bw": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 91, 95, 20, 21, 90, 94, 24, 25, 89, 93, 28, 29, 88, 92, 44, 40, 36, 32, 45, 41, 37, 33, 46, 42, 38, 34, 47, 43, 39, 35, 67, 71, 50, 51, 66, 70, 54, 55, 65, 69, 58, 59, 64, 68, 62, 63, 19, 23, 27, 31, 18, 22, 26, 30, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 49, 53, 57, 61, 48, 52, 56, 60]),
    "Bw'": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 68, 64, 20, 21, 69, 65, 24, 25, 70, 66, 28, 29, 71, 67, 35, 39, 43, 47, 34, 38, 42, 46, 33, 37, 41, 45, 32, 36, 40, 44, 92, 88, 50, 51, 93, 89, 54, 55, 94, 90, 58, 59, 95, 91, 62, 63, 60, 56, 52, 48, 61, 57, 53, 49, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 30, 26, 22, 18, 31, 27, 23, 19]),
    "Bw2": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 61, 60, 20, 21, 57, 56, 24, 25, 53, 52, 28, 29, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 50, 51, 27, 26, 54, 55, 23, 22, 58, 59, 19, 18, 62, 63, 95, 94, 93, 92, 91, 90, 89, 88, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 71, 70, 69, 68, 67, 66, 65, 64]),
    
    # inner moves
    "r": np.array([0, 1, 82, 3, 4, 5, 86, 7, 8, 9, 90, 11, 12, 13, 94, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 78, 34, 35, 36, 74, 38, 39, 40, 70, 42, 43, 44, 66, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 2, 67, 68, 69, 6, 71, 72, 73, 10, 75, 76, 77, 14, 79, 80, 81, 45, 83, 84, 85, 41, 87, 88, 89, 37, 91, 92, 93, 33, 95]),
    "r'": np.array([0, 1, 66, 3, 4, 5, 70, 7, 8, 9, 74, 11, 12, 13, 78, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 94, 34, 35, 36, 90, 38, 39, 40, 86, 42, 43, 44, 82, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 45, 67, 68, 69, 41, 71, 72, 73, 37, 75, 76, 77, 33, 79, 80, 81, 2, 83, 84, 85, 6, 87, 88, 89, 10, 91, 92, 93, 14, 95]),
    "r2": np.array([0, 1, 45, 3, 4, 5, 41, 7, 8, 9, 37, 11, 12, 13, 33, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 14, 34, 35, 36, 10, 38, 39, 40, 6, 42, 43, 44, 2, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 82, 67, 68, 69, 86, 71, 72, 73, 90, 75, 76, 77, 94, 79, 80, 81, 66, 83, 84, 85, 70, 87, 88, 89, 74, 91, 92, 93, 78, 95]),
    "l": np.array([0, 65, 2, 3, 4, 69, 6, 7, 8, 73, 10, 11, 12, 77, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 93, 35, 36, 37, 89, 39, 40, 41, 85, 43, 44, 45, 81, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 46, 66, 67, 68, 42, 70, 71, 72, 38, 74, 75, 76, 34, 78, 79, 80, 1, 82, 83, 84, 5, 86, 87, 88, 9, 90, 91, 92, 13, 94, 95]),
    "l'": np.array([0, 81, 2, 3, 4, 85, 6, 7, 8, 89, 10, 11, 12, 93, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 77, 35, 36, 37, 73, 39, 40, 41, 69, 43, 44, 45, 65, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 1, 66, 67, 68, 5, 70, 71, 72, 9, 74, 75, 76, 13, 78, 79, 80, 46, 82, 83, 84, 42, 86, 87, 88, 38, 90, 91, 92, 34, 94, 95]),
    "l2": np.array([0, 46, 2, 3, 4, 42, 6, 7, 8, 38, 10, 11, 12, 34, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 13, 35, 36, 37, 9, 39, 40, 41, 5, 43, 44, 45, 1, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 81, 66, 67, 68, 85, 70, 71, 72, 89, 74, 75, 76, 93, 78, 79, 80, 65, 82, 83, 84, 69, 86, 87, 88, 73, 90, 91, 92, 77, 94, 95]),
    "u": np.array([0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 36, 37, 38, 39, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 52, 53, 54, 55, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 4, 5, 6, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "u'": np.array([0, 1, 2, 3, 52, 53, 54, 55, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 20, 21, 22, 23, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 36, 37, 38, 39, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "u2": np.array([0, 1, 2, 3, 36, 37, 38, 39, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 52, 53, 54, 55, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 4, 5, 6, 7, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 20, 21, 22, 23, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "d": np.array([0, 1, 2, 3, 4, 5, 6, 7, 56, 57, 58, 59, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 40, 41, 42, 43, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "d'": np.array([0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 40, 41, 42, 43, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 56, 57, 58, 59, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 8, 9, 10, 11, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "d2": np.array([0, 1, 2, 3, 4, 5, 6, 7, 40, 41, 42, 43, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 56, 57, 58, 59, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 8, 9, 10, 11, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 24, 25, 26, 27, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]),
    "f": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 72, 18, 19, 20, 73, 22, 23, 24, 74, 26, 27, 28, 75, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 84, 51, 52, 53, 85, 55, 56, 57, 86, 59, 60, 61, 87, 63, 64, 65, 66, 67, 68, 69, 70, 71, 62, 58, 54, 50, 76, 77, 78, 79, 80, 81, 82, 83, 29, 25, 21, 17, 88, 89, 90, 91, 92, 93, 94, 95]),
    "f'": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 87, 18, 19, 20, 86, 22, 23, 24, 85, 26, 27, 28, 84, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 75, 51, 52, 53, 74, 55, 56, 57, 73, 59, 60, 61, 72, 63, 64, 65, 66, 67, 68, 69, 70, 71, 17, 21, 25, 29, 76, 77, 78, 79, 80, 81, 82, 83, 50, 54, 58, 62, 88, 89, 90, 91, 92, 93, 94, 95]),
    "f2": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 62, 18, 19, 20, 58, 22, 23, 24, 54, 26, 27, 28, 50, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 29, 51, 52, 53, 25, 55, 56, 57, 21, 59, 60, 61, 17, 63, 64, 65, 66, 67, 68, 69, 70, 71, 87, 86, 85, 84, 76, 77, 78, 79, 80, 81, 82, 83, 75, 74, 73, 72, 88, 89, 90, 91, 92, 93, 94, 95]),
    "b": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 91, 19, 20, 21, 90, 23, 24, 25, 89, 27, 28, 29, 88, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 71, 50, 51, 52, 70, 54, 55, 56, 69, 58, 59, 60, 68, 62, 63, 64, 65, 66, 67, 18, 22, 26, 30, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 49, 53, 57, 61, 92, 93, 94, 95]),
    "b'": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 68, 19, 20, 21, 69, 23, 24, 25, 70, 27, 28, 29, 71, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 88, 50, 51, 52, 89, 54, 55, 56, 90, 58, 59, 60, 91, 62, 63, 64, 65, 66, 67, 61, 57, 53, 49, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 30, 26, 22, 18, 92, 93, 94, 95]),
    "b2": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 61, 19, 20, 21, 57, 23, 24, 25, 53, 27, 28, 29, 49, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 30, 50, 51, 52, 26, 54, 55, 56, 22, 58, 59, 60, 18, 62, 63, 64, 65, 66, 67, 91, 90, 89, 88, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 71, 70, 69, 68, 92, 93, 94, 95]),
}


# if you want to define more moves or so
# def define_move(cycles):
#     ids = np.array(range(16*6))
#     ids_ref = np.array(range(16*6))

#     for cycle in cycles:
#         l = len(cycle)
#         ids[cycle[0]] = ids_ref[cycle[-1]]
#         for i in range(1,l):
#             ids[cycle[i]] = ids_ref[cycle[i-1]]
#     return ids

# cycles = [[2,66,45,82],[6,70,41,86],[10,74,37,90],[14,78,33,94]]

# print(f"np.array({define_move(cycles).tolist()})")



# jitted functions
@njit
def _change_state(state,changes):
    new_state = np.copy(state)
    for i,j in enumerate(changes):
        new_state[i] = state[j]   
    return new_state


class Cube4:
    def __init__(self, mode = 0):
        
        # standard mode, 1 = front color, 2 = right col, 3 = back col, 4 = left col, 5 = top col, 6 = bottom col
        self.state = np.array([1]*16+[2]*16+[3]*16+[4]*16+[5]*16+[6]*16)
        if mode == 1: # if the mode is 1, it shows the indexes instead of colors
            self.state = np.arange(16*6)
        
    def __repr__(self):
        str_repr = f"\n         {self.state[64]} {self.state[65]} {self.state[66]} {self.state[67]}"
        str_repr += f"\n         {self.state[68]} {self.state[69]} {self.state[70]} {self.state[71]}"
        str_repr += f"\n         {self.state[72]} {self.state[73]} {self.state[74]} {self.state[75]}"
        str_repr += f"\n         {self.state[76]} {self.state[77]} {self.state[78]} {self.state[79]}"
        str_repr += f"\n{self.state[48]} {self.state[49]} {self.state[50]} {self.state[51]}  {self.state[0]} {self.state[1]} {self.state[2]} {self.state[3]}  {self.state[16]} {self.state[17]} {self.state[18]} {self.state[19]}  {self.state[32]} {self.state[33]} {self.state[34]} {self.state[35]}"
        str_repr += f"\n{self.state[52]} {self.state[53]} {self.state[54]} {self.state[55]}  {self.state[4]} {self.state[5]} {self.state[6]} {self.state[7]}  {self.state[20]} {self.state[21]} {self.state[22]} {self.state[23]}  {self.state[36]} {self.state[37]} {self.state[38]} {self.state[39]}"
        str_repr += f"\n{self.state[56]} {self.state[57]} {self.state[58]} {self.state[59]}  {self.state[8]} {self.state[9]} {self.state[10]} {self.state[11]}  {self.state[24]} {self.state[25]} {self.state[26]} {self.state[27]}  {self.state[40]} {self.state[41]} {self.state[42]} {self.state[43]}"
        str_repr += f"\n{self.state[60]} {self.state[61]} {self.state[62]} {self.state[63]}  {self.state[12]} {self.state[13]} {self.state[14]} {self.state[15]}  {self.state[28]} {self.state[29]} {self.state[30]} {self.state[31]}  {self.state[44]} {self.state[45]} {self.state[46]} {self.state[47]}"
        str_repr += f"\n         {self.state[80]} {self.state[81]} {self.state[82]} {self.state[83]}"
        str_repr += f"\n         {self.state[84]} {self.state[85]} {self.state[86]} {self.state[87]}"
        str_repr += f"\n         {self.state[88]} {self.state[89]} {self.state[90]} {self.state[91]}"
        str_repr += f"\n         {self.state[92]} {self.state[93]} {self.state[94]} {self.state[95]}"
        
        return str_repr

    #to be defined
#     def plot(self,colors=["grey","green","r","b","darkorange","w","y"]):
#         plot_2d_cube(self.state, colors=colors) 
    
    def change_state(self,changes):
        self.state = _change_state(self.state,changes)
    
    def apply_moves(self,alg):
        # alg in string
        if alg:
            alg = alg.split(" ")
            for move in alg:
                self.change_state(move_dict[move])
                
    #to be defined, is nice if you want to apply a specific alg repeatedly
#     def apply_algs(self,alg_list):
#         # moves in string
#         if alg_list:
#             algs = alg_list.split(" ")
#             for alg in algs:
#                 self.change_state(alg_dict[alg])
    
cube = Cube4()
print(cube)
# time test:
alg = "Rw' U Rw' U' Rw2 R' U' Rw' U' R U2 Rw' U' Rw' U2 Rw' U2 Rw'"
t = time.time()
for i in range(100000):
    cube.apply_moves(alg)
print(time.time() - t)
print(cube)



