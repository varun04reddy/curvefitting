import math
import random

def mean(data):
    return sum(data) / len(data)

def variance(data):
    mu = mean(data)
    return sum((x - mu) ** 2 for x in data) / len(data)

def covariance(x, y):
    return sum((xi - mean(x)) * (yi - mean(y)) for xi, yi in zip(x, y)) / len(x)

def linear_least_squares(X, Y):
    b1 = covariance(X, Y) / variance(X)
    b0 = mean(Y) - b1 * mean(X)
    return b1, b0

def total_least_squares(X, Y):
    n = len(X)
    x_mean = mean(X)
    y_mean = mean(Y)
    st2 = sum((x - x_mean) ** 2 + (y - y_mean) ** 2 for x, y in zip(X, Y))
    beta = (sum((x - x_mean) * (y - y_mean) for x, y in zip(X, Y)) * -2 / st2) ** 0.5
    alpha = y_mean - beta * x_mean
    return beta, alpha

def ransac(X, Y, max_iterations=100, distance_threshold=5.0, inlier_ratio=0.6):
    best_slope = None
    best_intercept = None
    best_error = float('inf')

    for _ in range(max_iterations):
        samples = random.sample(list(zip(X, Y)), 2)
        x_samples, y_samples = zip(*samples)
        slope, intercept = linear_least_squares(x_samples, y_samples)
        inliers = sum(math.sqrt((slope * x + intercept - y) ** 2) < distance_threshold for x, y in zip(X, Y))

        if inliers > inlier_ratio * len(X):
            error = sum((slope * x + intercept - y) ** 2 for x, y in zip(X, Y))
            if error < best_error:
                best_error = error
                best_slope = slope
                best_intercept = intercept

    return best_slope, best_intercept

def calculate_eigenvectors(X, Y):
    x_var = variance(X)
    y_var = variance(Y)
    xy_cov = covariance(X, Y)
    T = x_var + y_var
    D = x_var * y_var - xy_cov ** 2
    eigenvalues = [((T + math.sqrt(T ** 2 - 4 * D)) / 2), ((T - math.sqrt(T ** 2 - 4 * D)) / 2)]
    eigenvectors = []

    for eig in eigenvalues:
        if xy_cov != 0:
            eigenvectors.append([-xy_cov / (x_var - eig), 1])
        else:
            if x_var - eig == 0:
                eigenvectors.append([1, 0])
            else:
                eigenvectors.append([0, 1])

    return eigenvectors
