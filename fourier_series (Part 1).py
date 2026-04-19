"""
Fourier Series Coefficients - Numerical Computation
====================================================
Part 1: Analytical function (numerical integration via Simpson's rule)
Part 2: Discrete time-series data (numerical sums)
Optional: FFT comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


# ─────────────────────────────────────────────
#  PART 1: Analytical Function
# ─────────────────────────────────────────────

def compute_fourier_analytical(f, L, N, num_points=1000):
    """
    Compute Fourier coefficients a0, A_n, B_n for an analytically defined
    function f over [0, L] using Simpson's rule.

    Parameters
    ----------
    f          : callable  — the function f(x)
    L          : float     — period / domain length
    N          : int       — number of Fourier modes
    num_points : int       — integration resolution (must be even)

    Returns
    -------
    a0   : float
    A    : ndarray of shape (N,)  — sine coefficients
    B    : ndarray of shape (N,)  — cosine coefficients
    """
    if num_points % 2 != 0:
        num_points += 1                     # Simpson's rule needs even intervals

    x = np.linspace(0, L, num_points + 1)  # num_points+1 sample points
    fx = f(x)

    # a0
    a0 = (1 / L) * simpson(fx, x=x)

    A = np.zeros(N)
    B = np.zeros(N)

    for n in range(1, N + 1):
        omega_n = 2 * np.pi * n / L

        integrand_A = fx * np.sin(omega_n * x)
        integrand_B = fx * np.cos(omega_n * x)

        A[n - 1] = (2 / L) * simpson(integrand_A, x=x)
        B[n - 1] = (2 / L) * simpson(integrand_B, x=x)

    return a0, A, B


def reconstruct_fourier(x, L, a0, A, B):
    """Reconstruct f_N(x) from Fourier coefficients."""
    N = len(A)
    result = np.full_like(x, a0, dtype=float)
    for n in range(1, N + 1):
        omega_n = 2 * np.pi * n / L
        result += A[n - 1] * np.sin(omega_n * x) + B[n - 1] * np.cos(omega_n * x)
    return result


def part1_demo():
    """Run Part 1 for two test functions and plot results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Part 1 — Fourier Coefficients from Analytical Function", fontsize=14)

    # ── Test function 1: f(x) = x on [0, 2π] ──────────────────────────────
    L1 = 2 * np.pi
    f1 = lambda x: x

    for col, N in enumerate([5, 20]):
        a0, A, B = compute_fourier_analytical(f1, L1, N)

        x_plot = np.linspace(0, L1, 500)
        f_exact = f1(x_plot)
        f_approx = reconstruct_fourier(x_plot, L1, a0, A, B)

        ax = axes[0, col]
        ax.plot(x_plot, f_exact,  'k-',  lw=2,   label='f(x) = x')
        ax.plot(x_plot, f_approx, 'r--', lw=1.5, label=f'Fourier (N={N})')
        ax.set_title(f"f(x)=x,  N={N}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        print(f"\n[f(x)=x, N={N}]  a0={a0:.4f}  "
              f"A[:3]={A[:3].round(4)}  B[:3]={B[:3].round(4)}")


if __name__ == "__main__":
    print("=" * 55)
    print(" PART 1: Fourier Coefficients — Analytical Function")
    print("=" * 55)
    part1_demo()

   
    