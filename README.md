# Gaussian Curve Fitting Project

## Overview
This project involves fitting a Gaussian curve to a set of data points using various mathematical techniques. The methods include basic gradient descent, the Levenberg-Marquardt algorithm, and Bayesian inference. The goal is to optimize the parameters of the Gaussian function to best fit the given data.

## Gaussian Function

$$
G(x, A, \mu, \sigma) = A \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

## Gradient Descent
$$
\text{grad}_A = -2 \sum (y - G(x, A, \mu, \sigma)) \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

$$
\text{grad}_\mu = -2 \sum (y - G(x, A, \mu, \sigma)) \frac{A(x - \mu)}{\sigma^2} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

$$
\text{grad}_\sigma = -2 \sum (y - G(x, A, \mu, \sigma)) \frac{A(x - \mu)^2}{\sigma^3} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

## Levenberg-Marquardt Algorithm
The Levenberg-Marquardt algorithm is a combination of gradient descent and the Gauss-Newton method. It minimizes the sum of squared errors by iteratively updating the parameters:

$$
\mathbf{J} = \frac{\partial r}{\partial \mathbf{p}}
$$

$$
\mathbf{H} = \mathbf{J}^T \mathbf{J}
$$

$$
\mathbf{g} = \mathbf{J}^T \mathbf{r}
$$

$$
\Delta \mathbf{p} = (\mathbf{H} + \lambda \mathbf{I})^{-1} \mathbf{g}
$$


## Bayesian Inference
Bayesian inference involves sampling parameter values from prior distributions and computing their posterior distributions based on the observed data:

$$
P(A, \mu, \sigma \mid x, y) \propto P(y \mid x, A, \mu, \sigma) P(A) P(\mu) P(\sigma)
$$


## Important Mathematical Concepts

### Least Squares and Gauss' Contribution
The method of least squares minimizes the sum of squared errors:

$$
\sum_{i=1}^n \epsilon_i^2
$$

$$
\Omega = \prod_{i=1}^n \phi(\epsilon_i)
$$

Gauss assumed that the best value to summarize measurements is the mean, which aligns with minimizing the sum of squared errors.

### Gaussian Distribution
Gauss concluded that the error distribution \(\phi\) should be symmetric and have its maximum at zero. He derived that the distribution must be:

$$
\phi(\epsilon_i) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{\epsilon_i^2}{2 \sigma^2}\right)
$$


### Maximizing Likelihood
The likelihood function for the Gaussian distribution is:

$$
\Omega = \prod_{i=1}^n \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{\epsilon_i^2}{2 \sigma^2}\right)
$$

Taking the logarithm simplifies the maximization to minimizing the sum of squared errors:

$$
\text{maximize} \quad -\sum_{i=1}^n \epsilon_i^2
$$

## Pros and Cons of Each Algorithm
Each algorithm used in Gaussian curve fitting has its pros and cons. Gradient Descent is simple to implement and effective for large datasets, but it can converge slowly and is sensitive to the choice of learning rate, making it suitable for straightforward optimization problems with large datasets. The Levenberg-Marquardt algorithm combines the benefits of gradient descent and Gauss-Newton methods, offering faster convergence for nonlinear least squares problems, but it is computationally intensive and requires good initial parameter estimates. It is ideal for problems requiring a balance between speed and accuracy. Bayesian Inference provides probabilistic parameter estimates and quantifies uncertainty but is computationally intensive and requires prior distributions. It is best for scenarios where understanding uncertainty and probabilistic nature of parameter estimates is crucial.
