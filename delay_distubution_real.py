import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from scipy.stats import gaussian_kde

# --- Load data ---
df = pd.read_csv("data_real.csv", sep=";")
df.columns = df.columns.str.strip()
delays = (df["Actual"] - df["Estimated"]).values.astype(float)

# --- Modified PERT distribution ---
# Parameterised by (a=min, m=mode, b=max, gamma=shape weight).
# gamma=4 is standard PERT; higher values pull the distribution tighter around m.

def pert_params(a, m, b, gamma=4):
    """Return (alpha1, alpha2) for the underlying Beta distribution."""
    alpha1 = 1 + gamma * (m - a) / (b - a)
    alpha2 = 1 + gamma * (b - m) / (b - a)
    return alpha1, alpha2

def pert_pdf(x, a, m, b, gamma=4):
    """Evaluate the PERT PDF at x given support [a, b]."""
    alpha1, alpha2 = pert_params(a, m, b, gamma)
    # Normalise x to [0, 1] for the Beta PDF, then scale density back
    z = (x - a) / (b - a)
    return beta_dist.pdf(z, alpha1, alpha2) / (b - a)

def pert_cdf(x, a, m, b, gamma=4):
    alpha1, alpha2 = pert_params(a, m, b, gamma)
    z = (x - a) / (b - a)
    return beta_dist.cdf(z, alpha1, alpha2)

def pert_mean(a, m, b, gamma=4):
    return (a + gamma * m + b) / (gamma + 2)

def pert_std(a, m, b, gamma=4):
    mu = pert_mean(a, m, b, gamma)
    return (b - a) / (gamma + 2) * np.sqrt((gamma + 2 + 1) / (gamma + 2 + 2))

# --- PERT parameters derived from the delay data ---
a = float(delays.min())
b = float(delays.max())

# Mode: bin the data and take the centre of the most populated bin
counts, edges = np.histogram(delays, bins="auto")
mode_est = float((edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2)

gamma = 80  # standard PERT weight

alpha1, alpha2 = pert_params(a, mode_est, b, gamma)

print("=== Delay statistics (Actual - Estimated) ===")
print(f"  n          = {len(delays)}")
print(f"  Min  (a)   = {a:.2f}")
print(f"  Mode (m)   = {mode_est:.2f}")
print(f"  Max  (b)   = {b:.2f}")
print(f"  Gamma      = {gamma}")
print(f"  Alpha1     = {alpha1:.4f}")
print(f"  Alpha2     = {alpha2:.4f}")
print(f"  PERT mean  = {pert_mean(a, mode_est, b, gamma):.2f}")
print(f"  PERT std   = {pert_std(a, mode_est, b, gamma):.2f}")
print(f"  Sample mean= {delays.mean():.2f}")
print(f"  Sample std = {delays.std():.2f}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Modified PERT Distribution of Task Delays (Actual − Estimated)", fontsize=13)

x = np.linspace(a - 1, b + 1, 500)
pdf_vals = pert_pdf(x, a, mode_est, b, gamma)

# Left panel: histogram + PERT PDF
ax1 = axes[0]
ax1.hist(delays, bins="auto", density=True, alpha=0.55, color="steelblue",
         edgecolor="white", label="Observed delays")
ax1.plot(x, pdf_vals, color="crimson", linewidth=2,
         label=f"Modified PERT (γ={gamma})\nm={mode_est:.1f}")

# KDE overlay for visual reference
kde = gaussian_kde(delays)
ax1.plot(x, kde(x), color="darkorange", linewidth=1.5, linestyle="--", label="KDE")

ax1.axvline(pert_mean(a, mode_est, b, gamma), color="crimson",
            linestyle=":", linewidth=1.2, label=f"PERT mean={pert_mean(a, mode_est, b, gamma):.1f}")
ax1.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="Zero delay")
ax1.set_xlabel("Delay (Actual − Estimated) [time units]")
ax1.set_ylabel("Density")
ax1.set_title("Histogram + Modified PERT PDF")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Right panel: CDF
ax2 = axes[1]
cdf_vals = pert_cdf(x, a, mode_est, b, gamma)
ax2.plot(x, cdf_vals, color="crimson", linewidth=2, label="PERT CDF")

# Empirical CDF
sorted_delays = np.sort(delays)
ecdf_y = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
ax2.step(sorted_delays, ecdf_y, color="steelblue", linewidth=1.5,
         alpha=0.8, label="Empirical CDF")

p50 = float(np.interp(0.50, cdf_vals, x))
p80 = float(np.interp(0.80, cdf_vals, x))
p90 = float(np.interp(0.90, cdf_vals, x))
for p, label in [(p50, "P50"), (p80, "P80"), (p90, "P90")]:
    ax2.axvline(p, linestyle=":", linewidth=1, alpha=0.7,
                label=f"{label}={p:.1f}")

ax2.set_xlabel("Delay (Actual − Estimated) [time units]")
ax2.set_ylabel("Cumulative probability")
ax2.set_title("Empirical vs PERT CDF")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("delay_distribution_real.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to delay_distribution_real.png")

# --- Gamma sweep: compare multiple gamma values against KDE ---
fig2, ax = plt.subplots(figsize=(9, 4))
fig2.suptitle("Gamma Sweep — Modified PERT vs KDE", fontsize=13)
ax.hist(delays, bins="auto", density=True, alpha=0.4, color="steelblue",
        edgecolor="white", label="Observed delays")
ax.plot(x, kde(x), color="black", linewidth=2, linestyle="--", label="KDE")
for g in [5, 20, 40, 60, 80]:
    ax.plot(x, pert_pdf(x, a, mode_est, b, g), label=f"γ={g}")
ax.set_xlabel("Delay (Actual − Estimated) [time units]")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gamma_sweep_real.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gamma sweep plot saved to gamma_sweep_real.png")
