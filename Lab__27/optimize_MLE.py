import numpy as np

## Scratch implementation

# Step 1: Simulate data
np.random.seed(42)
data = np.random.normal(loc=10, scale=3, size=1000)

# Step 2: Define the log-likelihood function (Normal distribution)
def log_likelihood(mu, sigma, data):
    n = len(data)
    if sigma <= 0:
        return -np.inf  # log-likelihood undefined for non-positive sigma
    ll = -n * np.log(np.sqrt(2 * np.pi) * sigma) - np.sum((data - mu)**2) / (2 * sigma**2)
    return ll

# Step 3: Implement simple gradient ascent to maximize the log-likelihood
def gradient_ascent(data, mu_init, sigma_init, learning_rate=0.001, iterations=1000):
    mu, sigma = mu_init, sigma_init
    n = len(data)

    for i in range(iterations):
        # Compute gradients manually
        dL_dmu = np.sum(data - mu) / (sigma**2)
        dL_dsigma = (-n / sigma) + (np.sum((data - mu)**2) / sigma**3)

        # Update parameters
        mu += learning_rate * dL_dmu
        sigma += learning_rate * dL_dsigma

        # Prevent sigma from going negative
        if sigma <= 0:
            sigma = 1e-3

    return mu, sigma

# Step 4: Run gradient ascent
mu_est, sigma_est = gradient_ascent(data, mu_init=0.0, sigma_init=1.0)

print(f"Estimated mu: {mu_est:.4f}")
print(f"Estimated sigma: {sigma_est:.4f}")


## using library

# from scipy.optimize import minimize
# from scipy.stats import norm
#
# # 1. Simulate data
# np.random.seed(42)  # for reproducibility
# data = np.random.normal(loc=10, scale=3, size=1000)
#
# # 2. Define negative log-likelihood function
# def neg_log_likelihood(params):
#     mu, sigma = params
#     if sigma <= 0:
#         return np.inf  # sigma must be positive
#     return -np.sum(norm.logpdf(data, loc=mu, scale=sigma))
#
# # 3. Initial guesses
# initial_guess = [0, 1]
#
# # 4. Optimize
# result = minimize(neg_log_likelihood, initial_guess, bounds=[(None, None), (1e-6, None)])
#
# # 5. Estimated parameters
# estimated_mu, estimated_sigma = result.x
#
# print(f"Estimated mu: {estimated_mu:.4f}")
# print(f"Estimated sigma: {estimated_sigma:.4f}")

