# Gaussian Curve Fitting Project


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


### Gaussian Distribution

$$
\phi(\epsilon_i) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{\epsilon_i^2}{2 \sigma^2}\right)
$$


### Maximizing Likelihood

$$
\Omega = \prod_{i=1}^n \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{\epsilon_i^2}{2 \sigma^2}\right)
$$


$$
\text{maximize} \quad -\sum_{i=1}^n \epsilon_i^2
$$

