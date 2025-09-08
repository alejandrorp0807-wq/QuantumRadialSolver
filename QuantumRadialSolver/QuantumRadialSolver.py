import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Constants
# ---------------------------
BOHR_RADIUS = 5.292e-11  # m
E_CHARGE = 1.6e-19       # C
EPS0 = 8.85e-12          # Vacuum permittivity
E0 = 13.6                # eV (ground state energy of hydrogen)

# Radial grid
UMIN = 0.01 * BOHR_RADIUS
UMAX = 100 * BOHR_RADIUS
NPTS = 1000
u = np.linspace(UMIN, UMAX, NPTS) / BOHR_RADIUS
du = u[1] - u[0]

# Bisection parameters
DELTA_ALPHA = 0.05
NREPS = 200


# ---------------------------
# Potentials
# ---------------------------
def hydrogen_potential(u, alpha, l):
    """Effective potential for Hydrogen atom."""
    return l * (l + 1) / u**2 - 2 / u + alpha


def hydrogen_like_potential(u, alpha, l):
    """Effective potential for Hydrogen-like atom."""
    if u > 3:
        c2 = -1 / (3 * u)
    else:
        c2 = -1 / u + 2 / 9
    return l * (l + 1) / u**2 + 2 * c2 + alpha


def lithium_potential(u, alpha, l):
    """Effective potential approximation for Lithium atom."""
    if u > 1:
        return l * (l + 1) / u**2 - 2 / (3 * u) + alpha
    else:
        return l * (l + 1) / u**2 - 2 / u + 4 / 9 + alpha


# ---------------------------
# Numerov method
# ---------------------------
def numerov(alpha, f, l):
    """Numerov integration to solve the radial Schrödinger equation."""
    psi = np.zeros(NPTS)
    phi = np.zeros(NPTS)

    # Initial conditions
    psi[0] = 1e-2
    psi[1] = 1e-1
    phi[0] = psi[0] * (1 - du**2 * f(u[0], alpha, l) / 12)
    phi[1] = psi[1] * (1 - du**2 * f(u[1], alpha, l) / 12)

    for i in range(1, NPTS - 1):
        phi[i + 1] = 2 * phi[i] - phi[i - 1] + du**2 * f(u[i], alpha, l) * psi[i]
        psi[i + 1] = phi[i + 1] / (1 - du**2 * f(u[i + 1], alpha, l) / 12)

    return psi[-1]


def numerov_solution(alpha, f, l):
    """Returns the full radial wavefunction using Numerov integration."""
    psi = np.zeros(NPTS)
    phi = np.zeros(NPTS)

    psi[0] = 1e-2
    psi[1] = 1e-1
    phi[0] = psi[0] * (1 - du**2 * f(u[0], alpha, l) / 12)
    phi[1] = psi[1] * (1 - du**2 * f(u[1], alpha, l) / 12)

    for i in range(1, NPTS - 1):
        phi[i + 1] = 2 * phi[i] - phi[i - 1] + du**2 * f(u[i], alpha, l) * psi[i]
        psi[i + 1] = phi[i + 1] / (1 - du**2 * f(u[i + 1], alpha, l) / 12)

    return psi


# ---------------------------
# Root-finding (Bisection)
# ---------------------------
def bisection(alpha0, nmax, f, l):
    """Find eigenvalues (alpha) using bisection + Numerov method."""
    n = 1
    solutions = []
    alpha1 = alpha0
    alpha2 = alpha0 + DELTA_ALPHA

    while n <= nmax:
        while numerov(alpha1, f, l) * numerov(alpha2, f, l) > 0:
            alpha1 = alpha2
            alpha2 += DELTA_ALPHA

        for _ in range(NREPS):
            alpha_new = (alpha1 + alpha2) / 2
            if numerov(alpha1, f, l) * numerov(alpha_new, f, l) < 0:
                alpha2 = alpha_new
            else:
                alpha1 = alpha_new

        if (alpha1 + alpha2) / 2 > 1:
            break

        solutions.append((alpha1 + alpha2) / 2)
        alpha1 = alpha2
        alpha2 = alpha1 + DELTA_ALPHA
        n += 1

    return np.array(solutions)


# ---------------------------
# Results & Plotting
# ---------------------------
def plot_results(f, l, case_name, alphas, num_states=4):
    """Plot radial wavefunctions and compare numerical vs analytical energies."""
    energies_numerical = []
    energies_analytical = []

    for n in range(num_states):
        psi = numerov_solution(alphas[min(n, len(alphas) - 1)], f, l)
        energy_num = -E0 / (n + alphas[min(n, len(alphas) - 1)])**2
        energies_numerical.append(energy_num)

        if (n + 1 - l) != 0:
            energy_analytic = -E0 / (n + 1 - l)**2
            energies_analytical.append(energy_analytic)

        plt.figure(figsize=(8, 5))
        plt.plot(u, psi, label=f"n={n+1}, {case_name}")
        plt.xlabel("Radial distance (Bohr units)")
        plt.ylabel("Radial wavefunction")
        plt.title(f"Radial wavefunction for {case_name}, n={n+1}")
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f"\nCase: {case_name}")
    print("Numerical energies (eV):", energies_numerical)
    print("Analytical energies (eV):", energies_analytical)


# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    # Hydrogen, l=0
    sol_h_10 = bisection(0, 4, hydrogen_potential, 0)
    print("Hydrogen (l=0) alphas:", sol_h_10)
    plot_results(hydrogen_potential, 0, "Hydrogen l=0", sol_h_10)

    # Hydrogen, l=1
    sol_h_11 = bisection(0, 4, hydrogen_potential, 1)
    print("Hydrogen (l=1) alphas:", sol_h_11)
    plot_results(hydrogen_potential, 1, "Hydrogen l=1", sol_h_11)

    # Hydrogen-like, l=0
    sol_hlike_20 = bisection(0, 4, hydrogen_like_potential, 0)
    print("Hydrogen-like (l=0) alphas:", sol_hlike_20)
    plot_results(hydrogen_like_potential, 0, "Hydrogen-like l=0", sol_hlike_20)

    # Lithium, l=0
    sol_li_30 = bisection(0, 4, lithium_potential, 0)
    print("Lithium (l=0) alphas:", sol_li_30)
    plot_results(lithium_potential, 0, "Lithium l=0", sol_li_30)
