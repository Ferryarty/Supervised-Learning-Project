# NSPT simulation for OU process (order 1) and comparison with analytics
import numpy as np
import matplotlib.pyplot as plt
# Parameters
b_true = 1.44 # true parameter (and we'll also use as b_star)
b_star = b_true # expansion point
dt = 0.01
N_steps = 300_000 # steps (long to get good statistics)
burn_fraction = 0.2
burn = int(burn_fraction * N_steps)
np.random.seed(42)
# Arrays for order-0 and order-1
x0 = np.zeros(N_steps)
x1 = np.zeros(N_steps)
x0[0] = 0.0
x1[0] = 0.0
# Euler-Maruyama updates for the tower:
# dx0 = -b_star * x0 dt + sqrt(2*dt)*xi
# dx1 = -b_star * x1 dt - x0 dt (deterministic, driven by x0)
for t in range(1, N_steps):
    xi = np.random.randn() # N(0,1)
    x0[t] = x0[t-1] - b_star * x0[t-1] * dt + np.sqrt(2 * dt) * xi
    x1[t] = x1[t-1] - b_star * x1[t-1] * dt - x0[t-1] * dt
# Discard burn-in
x0_ss = x0[burn:]
x1_ss = x1[burn:]
time = np.arange(len(x0_ss)) * dt
# Stationary estimates from NSPT tower
V0_est = np.mean(x0_ss**2)
C01_est = np.mean(x0_ss * x1_ss)
deriv_est = 2.0 * C01_est # d/db <x^2> at b_star
# Analytic values at b_star
V0_analytic = 1.0 / b_star
deriv_analytic = -1.0 / (b_star**2)
# Now check Taylor reconstruction: choose a small delta and compare to direct simulation
eps = 0.05 # perturbation in b
b_eps = b_star + eps
# Taylor reconstructed variance at b_eps
var_taylor = V0_est + 2 * eps * C01_est # up to linear order
# Direct simulation at b_eps (simple EM simulation) to compare
def simulate_ou_variance(b, Nsteps=200_000, dt=dt, burn_frac=0.2):
    rng = np.random.RandomState(123) # fixed seed for repeatability
    x = np.zeros(Nsteps)
    for t in range(1, Nsteps):
        x[t] = x[t-1] - b * x[t-1] * dt + np.sqrt(2 * dt) * rng.randn()
    burn = int(burn_frac * Nsteps)
    return np.mean(x[burn:]**2)
var_direct_eps = simulate_ou_variance(b_eps, Nsteps=200_000, dt=dt, burn_frac=0.2)
var_direct_star = simulate_ou_variance(b_star, Nsteps=200_000, dt=dt, burn_frac=0.2)
# Print results
print("NSPT estimates (from tower):")
print(f" V0_est = {V0_est:.6f} (analytic 1/b = {V0_analytic:.6f})")
print(f" C01_est = {C01_est:.6f}")
print(f" deriv_est = 2*C01 = {deriv_est:.6f} (analytic = {deriv_analytic:.6f})")
print()
print(f"Taylor linear reconstruction of var at b = b_star + eps (eps={eps}): {var_taylor:.6f}")
print(f"Direct sim var at b = {b_eps:.6f} : {var_direct_eps:.6f}")
print(f"Direct sim var at b = {b_star:.6f} (for reference): {var_direct_star:.6f}")
print(f"Analytic var at b = {b_eps:.6f} : {1.0/b_eps:.6f}")
# Quick diagnostic plots
fig, axes = plt.subplots(2,1, figsize=(9,6), sharex=True)
axes[0].plot(time[:2000], x0_ss[:2000], lw=0.8)
axes[0].set_ylabel("x^(0) (order 0)")
axes[0].grid(alpha=0.3)
axes[1].plot(time[:2000], x1_ss[:2000], lw=0.8, color='C1')
axes[1].set_ylabel("x^(1) (order 1)")
axes[1].set_xlabel("time")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()
# Histogram comparison: distribution of x0 and approximate x at perturbed b via linearization
# approximate x(b_eps) ≈ x0 + eps * x1
x_approx_eps = x0_ss + eps * x1_ss
fig, ax = plt.subplots(1,2, figsize=(11,4))
ax[0].hist(x0_ss, bins=80, density=True, alpha=0.6)
ax[0].set_title("Distribution of x^(0) (b=b_star)")
ax[1].hist(x_approx_eps, bins=80, density=True, alpha=0.6, label="NSPT linear approx")
# direct sim sample for b_eps to overlay
# generate a shorter sample for visualization
x_direct = np.zeros(50_000)
rng = np.random.RandomState(999)
for t in range(1, len(x_direct)):
    x_direct[t] = x_direct[t-1] - b_eps * x_direct[t-1] * dt + np.sqrt(2 * dt) * rng.randn()
ax[1].hist(x_direct[int(0.2*len(x_direct)):], bins=80, density=True, alpha=0.5, label="direct sim")
ax[1].legend()
ax[1].set_title("Approx x(b_star+eps) via NSPT linear vs direct sim")
plt.tight_layout()
plt.show()