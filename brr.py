import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Parameters for the log-normal distribution
mu = 0       # Mean of the underlying normal distribution
sigma = 0.5  # Standard deviation of the underlying normal distribution

# Create a log-normal distribution
s = sigma
x = np.linspace(0.01, 5, 1000)
pdf = lognorm.pdf(x, s, scale=np.exp(mu))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, pdf, color='steelblue', linewidth=2)
plt.title('Log-normal Distribution', fontsize=14)
plt.xlabel('Discharge', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
