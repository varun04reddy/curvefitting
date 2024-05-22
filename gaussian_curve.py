import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def generate_data(num_points):
    x = np.linspace(-5, 5, num_points)
    y = gaussian(x, 1, 0, 1) + 0.1 * np.random.normal(size=x.size)
    return x, y

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def fit_gaussian(x, y):
    amp_init = np.max(y)
    mean_init = x[np.argmax(y)]
    stddev_init = np.std(x)
    params = np.array([amp_init, mean_init, stddev_init])
    learning_rate = 1e-4
    num_iterations = 10000

    for _ in range(num_iterations):
        amp, mean, stddev = params
        y_pred = gaussian(x, amp, mean, stddev)
        error = y - y_pred
        grad_amp = -2 * np.sum(error * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)))
        grad_mean = -2 * np.sum(error * amp * (x - mean) * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) / stddev ** 2)
        grad_stddev = -2 * np.sum(error * amp * (x - mean) ** 2 * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) / stddev ** 3)
        params -= learning_rate * np.array([grad_amp, grad_mean, grad_stddev])

    return params


def jacobian(x, amp, mean, stddev):
    d_amp = np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    d_mean = amp * (x - mean) / (stddev ** 2) * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    d_stddev = amp * ((x - mean) ** 2) / (stddev ** 3) * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    return np.vstack([d_amp, d_mean, d_stddev]).T


def Levenberg_Marquardt(x,y):
    amp_init = np.max(y)
    mean_init = x[np.argmax(y)]
    stddev_init = np.std(x)
    params = np.array([amp_init, mean_init, stddev_init])
    num_iterations = 100
    lambda_val = 1e-2

    for _ in range(num_iterations):
        amp, mean, stddev = params
        y_pred = gaussian(x, amp, mean, stddev)
        error = y - y_pred
        jacobian_matrix = jacobian(x, amp, mean, stddev)
        hessian_matrix = jacobian_matrix.T @ jacobian_matrix
        params -= np.linalg.inv(hessian_matrix + lambda_val * np.eye(3)) @ jacobian_matrix.T @ error

    return params


def bayesian_gaussian_fit(x, y, num_samples=10000):
    amp_samples = np.random.uniform(0.5, 1.5, num_samples)
    mean_samples = np.random.uniform(-1, 1, num_samples)
    stddev_samples = np.random.uniform(0.5, 1.5, num_samples)

    likelihoods = []
    for amp, mean, stddev in zip(amp_samples, mean_samples, stddev_samples):
        y_pred = gaussian(x, amp, mean, stddev)
        likelihood = np.exp(-0.5 * np.sum((y - y_pred) ** 2))
        likelihoods.append(likelihood)

    likelihoods = np.array(likelihoods)
    likelihoods /= np.sum(likelihoods)

    amp_mean = np.sum(amp_samples * likelihoods)
    mean_mean = np.sum(mean_samples * likelihoods)
    stddev_mean = np.sum(stddev_samples * likelihoods)

    return amp_mean, mean_mean, stddev_mean

def plot_distance(x, y, amp, mean, stddev):
    y_pred = gaussian(x, amp, mean, stddev)
    distances = np.abs(y - y_pred)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data', color='red')
    plt.plot(x, gaussian(x, amp, mean, stddev), label='Fitted curve', color='blue')
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], y_pred[i]], color='green')
    plt.title('Distance from Points to Curve')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

def main():
    num_points = int(input("Num points to generate: "))
    x_data, y_data = generate_data(num_points)
    fitted_params = fit_gaussian(x_data, y_data)
    amp, mean, stddev = fitted_params

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Data', color='red')
    plt.plot(x_data, gaussian(x_data, amp, mean, stddev), label='Fitted curve', color='blue')
    plt.title('Gaussian Curve Fitting')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

    residuals = y_data - gaussian(x_data, amp, mean, stddev)
    plt.figure(figsize=(10, 4))
    plt.scatter(x_data, residuals, color='purple')
    plt.hlines(0, xmin=x_data.min(), xmax=x_data.max(), linestyles='dashed')
    plt.title('Residuals of the Fit')
    plt.xlabel('X-axis')
    plt.ylabel('Residuals')
    plt.show()

    print(f"Fitted parameters: Amplitude={amp}, Mean={mean}, Standard Deviation={stddev}")

    plot_distance(x_data, y_data, amp, mean, stddev)

if __name__ == "__main__":
    main()
