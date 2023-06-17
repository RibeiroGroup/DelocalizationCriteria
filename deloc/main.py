from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from mpmath import *
from numpy import pi
from sympy import erfc, exp
from mpmath import findroot, pi
from scipy.optimize import root

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


# ------------------------- Core Functions ---------------------------

def energy_cavity(m, x, Ec, Lc):
    """Calculate cavity photon dispersion."""
    return Ec * np.sqrt(1 + ((x * Lc) / (m * pi)) ** 2)


def energy_lp_zero(m, x, Ec, E0, Lc, Delta):
    """Calculate the zeroth-order approximation for lower polariton energies.

    :param E0: Mean molecular transition energy
    :param Delta: Half of the Rabi Splitting
    :param Lc: Cavity Length
    :param x: wave-vector number
    :type Ec: Cavity cut-off energy

    """
    result: Any = (E0 + energy_cavity(m, x, Ec, Lc)) / 2 - np.sqrt(
        Delta ** 2 + ((E0 - energy_cavity(m, x, Ec, Lc)) / 2) ** 2)
    return result


def energy_up_zero(m, x, Ec, E0, Lc, Delta):
    """Calculate the zeroth-order approximation for upper polariton energies."""
    result: Any = (E0 + energy_cavity(m, x, Ec, Lc)) / 2 + np.sqrt(
        Delta ** 2 + ((E0 - energy_cavity(m, x, Ec, Lc)) / 2) ** 2)
    return result


# ------------------- Electronic and Photonic Weights -----------------------

def c_ex_lp(m, x, Ec, E0, Lc, Delta):
    """Calculate the total electronic weight in lower polaritonic state."""
    return (Delta ** 2) / (Delta ** 2 + (energy_lp_zero(m, x, Ec, E0, Lc, Delta) - E0) ** 2)


def c_ex_up(m, x, Ec, E0, Lc, Delta):
    """Calculate the total electronic weight in upper polaritonic state."""
    return Delta ** 2 / (Delta ** 2 + (energy_up_zero(m, x, Ec, E0, Lc, Delta) - E0) ** 2)


def c_l_lp(m, x, Ec, E0, Lc, Delta):
    """Calculate the total photonic weight in lower polaritonic state."""
    return 1 - c_ex_lp(m, x, Ec, E0, Lc, Delta)


def c_l_up(m, x, Ec, E0, Lc, Delta):
    """Calculate the total photonic weight in upper polaritonic state."""
    return 1 - c_ex_up(m, x, Ec, E0, Lc, Delta)


# ----------------------- Group Velocity ---------------------------

def group_velocity_lp(m, x, Ec, E0, Lc, Delta):
    """Calculate group velocity of lower polariton."""
    return ((Ec ** 2) * (Lc ** 2) * x * c_l_lp(m, x, Ec, E0, Lc, Delta)) / (
            (energy_cavity(m, x, Ec, Lc)) * (pi ** 2) * (m ** 2))


def group_velocity_up(m, x, Ec, E0, Lc, Delta):
    """Calculate group velocity of upper polariton."""
    return ((Ec ** 2) * (Lc ** 2) * x * c_l_up(m, x, Ec, E0, Lc, Delta)) / (
            (energy_cavity(m, x, Ec, Lc)) * (pi ** 2) * (m ** 2))


# ----------------- Inelastic Energy Broadening ---------------------

def gamma_inel_lp(m, x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L):
    """Calculate inelastic energy broadening of lower polariton."""
    return c_ex_lp(m, x, Ec, E0, Lc, Delta) * Gamma_Ex + c_l_lp(m, x, Ec, E0, Lc, Delta) * Gamma_L


def gamma_inel_up(m, x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L):
    """Calculate inelastic energy broadening of upper polariton."""
    return c_ex_up(m, x, Ec, E0, Lc, Delta) * Gamma_Ex + c_l_up(m, x, Ec, E0, Lc, Delta) * Gamma_L


# ----------------- Inelastic Mean Free Path ---------------------

def l_inel_lp(m, x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L):
    """Calculate inelastic mean free path of lower polariton."""
    return group_velocity_lp(m, x, Ec, E0, Lc, Delta) / gamma_inel_lp(m, x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L)


def l_inel_up(m, x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L):
    """Calculate inelastic mean free path of upper polariton."""
    return group_velocity_up(m, x, Ec, E0, Lc, Delta) / gamma_inel_up(m, x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L)


# ----------------- Inelastic Boundary Points ---------------------

def q_inel_lp(m, x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L):
    return l_inel_lp(m, x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L) - 1 / x


def q_inel_up(m, x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L):
    return l_inel_up(m, x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L) - 1 / x


# ----------------- Energy Broadening due to Fluctuations ---------------------

def gamma_fluc_lp(m, x, Ec, E0, Lc, Delta, a):
    """Calculate lower polariton energy broadening due to fluctuations."""
    return ((3 * (m ** 2) * (pi ** 2)) / 2) * ((a / Lc) ** 3) * ((Delta ** 2) / Ec) * c_ex_lp(m, x, Ec, E0, Lc, Delta)


def gamma_fluc_up(m, x, Ec, E0, Lc, Delta, a):
    """Calculate upper polariton energy broadening due to fluctuations."""
    return ((3 * (m ** 2) * (pi ** 2)) / 2) * ((a / Lc) ** 3) * ((Delta ** 2) / Ec) * c_ex_up(m, x, Ec, E0, Lc, Delta)


# ----------------- Mean Free Path of Scattering by Fluctuations ---------------------

def l_fluc_lp(m, x, Ec, E0, Lc, Delta, a):
    """Calculate lower polariton mean free path of scattering by fluctuations."""
    return group_velocity_lp(m, x, Ec, E0, Lc, Delta) / gamma_fluc_lp(m, x, Ec, E0, Lc, Delta, a)


def l_fluc_up(m, x, Ec, E0, Lc, Delta, a):
    """Calculate upper polariton mean free path of scattering by fluctuations."""
    return group_velocity_up(m, x, Ec, E0, Lc, Delta) / gamma_fluc_up(m, x, Ec, E0, Lc, Delta, a)


# ----------------- Fluctuations Boundary Points ---------------------

def q_fluc_lp(m, x, Ec, E0, Lc, Delta, a):
    return l_fluc_lp(m, x, Ec, E0, Lc, Delta, a) - 1 / x


def q_fluc_up(m, x, Ec, E0, Lc, Delta, a):
    return l_fluc_up(m, x, Ec, E0, Lc, Delta, a) - 1 / x


# ----------------- Energy Broadening due to Resonant Scattering LP---------------------

# fadeeva function
# exp*erfc
def f(x):
    return exp(-(x ** 2)) * erfc(-1j * x)


def gamma_res_lp(m, x, Ec, E0, Lc, Delta, Sigma, imag_part_lp):
    """Calculate lower polariton energy broadening due to resonant scattering."""

    B = ((Delta / Sigma) ** 2) * np.sqrt(np.pi)

    # Prompt the user for the real and imaginary parts of the initial guess
    real_part_lp = energy_lp_zero(m, x, Ec, E0, Lc, Delta)

    # Create the initial guess as a complex number
    initial_guess_lp = complex(real_part_lp, imag_part_lp)

    t = findroot(lambda y: 1j * f((y - E0) / Sigma) * B * Sigma + y - Ec * np.sqrt(1 + ((x * Lc) / pi) ** 2),
                 initial_guess_lp)
    return t.imag


def gamma_res_lp2(m, x, Ec, E0, Lc, Delta, Sigma, imag_part_lp):
    """Calculate lower polariton energy broadening due to resonant scattering."""

    B = ((Delta / Sigma) ** 2) * np.sqrt(np.pi)

    # Prompt the user for the real and imaginary parts of the initial guess
    real_part_lp = energy_lp_zero(m, x, Ec, E0, Lc, Delta)

    # Create the initial guess as a complex number
    initial_guess_lp = complex(real_part_lp, imag_part_lp)

    # Define the f function

    # Define the equation to find the root
    def equation(y):
        return 1j * f((y - E0) / Sigma) * B * Sigma + y - Ec * np.sqrt(1 + ((x * Lc) / np.pi) ** 2)

    # Find the root using scipy.optimize.root
    result = findroot(equation, initial_guess_lp)
    t = result.x

    return t.imag


# ----------------- Energy Broadening due to Resonant Scattering UP ---------------------

def gamma_res_up(m, x, Ec, E0, Lc, Delta, Sigma, imag_part_up):
    """Calculate upper polariton energy broadening due to resonant scattering."""

    B = ((Delta / Sigma) ** 2) * np.sqrt(np.pi)

    # Prompt the user for the real and imaginary parts of the initial guess
    real_part_up = energy_up_zero(m, x, Ec, E0, Lc, Delta)

    # Create the initial guess as a complex number
    initial_guess_up = complex(real_part_up, imag_part_up)
    t = findroot(lambda y: 1j * f((y - E0) / Sigma) * B * Sigma + y - Ec * np.sqrt(1 + ((x * Lc) / pi) ** 2),
                 initial_guess_up)
    return t.imag


# ----------------- Boundary Points due to Inelastic Scattering ---------------------

def inelastic(m, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L, q_initial, q_final, q_step):
    q_domain = np.arange(q_initial, q_final, q_step)

    q_inel_array_lp = np.empty(len(q_domain), dtype=float)

    for i in range(len(q_domain)):
        q_inel_array_lp[i] = q_inel_lp(m, q_domain[i], Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L)

    q_inel_array_up = np.empty(len(q_domain), dtype=float)

    for i in range(len(q_domain)):
        q_inel_array_up[i] = q_inel_up(m, q_domain[i], Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L)

    # Calculate the differences between consecutive elements
    diff_arr_inel_lp = np.diff(np.sign(q_inel_array_lp))

    # Find the indices where the sign changes
    indices_inel_lp = np.where(diff_arr_inel_lp != 0)[0] + 1

    if len(indices_inel_lp) == 1:
        q_inel_lp_min = q_domain[indices_inel_lp[0]]  # Assign the element to q_min
        q_inel_lp_max = q_final
    else:
        q_inel_lp_min = q_domain[indices_inel_lp[0]]
        q_inel_lp_max = q_domain[indices_inel_lp[1]]  # Assign the second element to q_max

    # Calculate the differences between consecutive elements
    diff_arr_inel_up = np.diff(np.sign(q_inel_array_up))

    # Find the indices where the sign changes
    indices_inel_up = np.where(diff_arr_inel_up != 0)[0] + 1

    if len(indices_inel_up) == 1:
        q_inel_up_min = q_domain[indices_inel_up[0]]  # Assign the element to q_min
        q_inel_up_max = q_final
    else:
        q_inel_up_min = q_domain[indices_inel_up[0]]
        q_inel_up_max = q_domain[indices_inel_up[1]]  # Assign the second element to q_max

    return q_inel_lp_min, q_inel_lp_max, q_inel_up_min, q_inel_up_max


# ----------------- Boundary Points due to Fluctuations ---------------------

def fluctuation(m, Ec, E0, Lc, Delta, a, q_initial, q_final, q_step):
    q_domain = np.arange(q_initial, q_final, q_step)

    q_fluc_array_lp = np.empty(len(q_domain), dtype=float)

    for i in range(len(q_domain)):
        q_fluc_array_lp[i] = q_fluc_lp(m, q_domain[i], Ec, E0, Lc, Delta, a)

    q_fluc_array_up = np.empty(len(q_domain), dtype=float)

    for i in range(len(q_domain)):
        q_fluc_array_up[i] = q_fluc_up(m, q_domain[i], Ec, E0, Lc, Delta, a)

    # Calculate the differences between consecutive elements
    diff_arr_fluc_lp = np.diff(np.sign(q_fluc_array_lp))

    # Find the indices where the sign changes
    indices_fluc_lp = np.where(diff_arr_fluc_lp != 0)[0] + 1

    if len(indices_fluc_lp) == 1:
        q_fluc_lp_min = q_domain[indices_fluc_lp[0]]  # Assign the element to q_min
        q_fluc_lp_max = q_final
    else:
        q_fluc_lp_min = q_domain[indices_fluc_lp[0]]
        q_fluc_lp_max = indices_fluc_lp[1]  # Assign the second element to q_max

    # Calculate the differences between consecutive elements
    diff_arr_fluc_up = np.diff(np.sign(q_fluc_array_up))

    # Find the indices where the sign changes
    indices_fluc_up = np.where(diff_arr_fluc_up != 0)[0] + 1

    if len(indices_fluc_up) == 1:
        q_fluc_up_min = q_domain[indices_fluc_up[0]]  # Assign the element to q_min
        q_fluc_up_max = q_final
    else:
        q_fluc_up_min = q_domain[indices_fluc_up[0]]
        q_fluc_up_max = q_domain[indices_fluc_up[1]]  # Assign the second element to q_max

    return q_fluc_lp_min, q_fluc_lp_max, q_fluc_up_min, q_fluc_up_max


# ----------------- Boundary Points due to Resonant Scattering ---------------------

def resonant(m, Ec, E0, Lc, Delta, Sigma, q_initial, q_final, q_step):
    q_domain = np.arange(q_initial, q_final, q_step)

    # Initialize the arrays with the q_domain shape

    gamma_res_array_lp = np.empty_like(q_domain)
    l_res_array_lp = np.empty_like(q_domain)
    q_res_array_lp = np.empty_like(q_domain)

    # Calculate the arrays element-wise
    for i in range(len(q_domain) - 1):
        gamma_res_array_lp[0] = gamma_res_lp(m, q_domain[0], Ec, E0, Lc, Delta, Sigma, 0)
        gamma_res_array_lp[i + 1] = gamma_res_lp(m, q_domain[i + 1], Ec, E0, Lc, Delta, Sigma, gamma_res_array_lp[i])

    # ----------------- Mean Free Path of Resonant Scattering ---------------------
    for i in range(len(q_domain)):
        l_res_array_lp[i] = group_velocity_lp(m, q_domain[i], Ec, E0, Lc, Delta) / abs(gamma_res_array_lp[i])

    # ----------------- Resonant Boundary Points ---------------------
    for i in range(len(q_domain)):
        q_res_array_lp[i] = l_res_array_lp[i] - 1 / q_domain[i]

    # Calculate the differences between consecutive elements
    diff_arr_res_lp = np.diff(np.sign(q_res_array_lp))

    # Find the indices where the sign changes
    indices_res_lp = np.where(diff_arr_res_lp != 0)[0] + 1

    if len(indices_res_lp) == 1:
        q_res_lp_min = q_domain[indices_res_lp[0]]  # Assign the element to q_min
        q_res_lp_max = q_final
    else:
        q_res_lp_min = q_domain[indices_res_lp[0]]
        q_res_lp_max = q_domain[indices_res_lp[1]]  # Assign the second element to q_max

    # Initialize the arrays with the q_domain shape
    gamma_res_array_up = np.empty_like(q_domain)
    l_res_array_up = np.empty_like(q_domain)
    q_res_array_up = np.empty_like(q_domain)

    # Calculate the arrays element-wise
    for i in range(len(q_domain) - 1):
        gamma_res_array_up[0] = gamma_res_up(m, q_domain[0], Ec, E0, Lc, Delta, Sigma, 0)
        gamma_res_array_up[i + 1] = gamma_res_up(m, q_domain[i + 1], Ec, E0, Lc, Delta, Sigma, gamma_res_array_up[i])

        # Check if the current element is too small
        if abs(gamma_res_array_up[i + 1]) < 1e-6:
            # Make the rest of the array zero
            gamma_res_array_up[i + 2:] = 0
            # Exit the loop
            break

    # ----------------- Mean Free Path of Resonant Scattering ---------------------
    for i in range(len(q_domain)):
        l_res_array_up[i] = group_velocity_up(m, q_domain[i], Ec, E0, Lc, Delta) / abs(gamma_res_array_up[i])

    # ----------------- Resonant Boundary Points ---------------------
    for i in range(len(q_domain)):
        q_res_array_up[i] = l_res_array_up[i] - 1 / q_domain[i]

    # Calculate the differences between consecutive elements
    diff_arr_res_up = np.diff(np.sign(q_res_array_up))

    # Find the indices where the sign changes
    indices_res_up = np.where(diff_arr_res_up != 0)[0] + 1

    q_res_up_min = q_domain[indices_res_up[0]]  # Assign the element to q_min

    return q_res_lp_min, q_res_lp_max, q_res_up_min


# ----------------- Boundary Points Comparison ---------------------

def boundary(q_initial, q_final, q_step, m, E0, Ec, Lc, Delta, Sigma, Gamma_Ex, Gamma_L, a):
    q_domain = np.arange(q_initial, q_final, q_step)

    q_inel_lp_min, q_inel_lp_max, q_inel_up_min, _ = inelastic(m, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L, q_initial,
                                                               q_final, q_step)

    q_fluc_lp_min, q_fluc_lp_max, q_fluc_up_min, _ = fluctuation(m, Ec, E0, Lc, Delta, a, q_initial, q_final, q_step)

    q_res_lp_min, q_res_lp_max, q_res_up_min = resonant(m, Ec, E0, Lc, Delta, Sigma, q_initial, q_final, q_step)

    # Compare three scatterings to determine the minimum boundary for UP

    q_min_up = max(q_fluc_up_min, q_inel_up_min, q_res_up_min)

    # Compare three scatterings to determine the minimum boundary for LP

    q_min_lp = max(q_fluc_lp_min, q_inel_lp_min, q_res_lp_min)

    # Compare three scatterings to determine the maximum boundary for LP

    q_max_lp = min(q_fluc_lp_max, q_inel_lp_max, q_res_lp_max)

    return q_min_lp, q_max_lp, q_min_up


# ----------------- Wave-number vs Detuning Phase Diagram with no energetic disorder---------------------

def plot_q_vs_detuning_inel(q_initial, q_final, q_step, Ec_initial, Ec_final, Ec_step, m, E0, Lc, Delta, Gamma_Ex,
                            Gamma_L):
    q_domain = np.arange(q_initial, q_final, q_step)
    Ec_domain = np.arange(Ec_initial, Ec_final, Ec_step)

    q_inel_min_lp_values = []
    q_inel_max_lp_values = []
    Ec_values = []
    detuning_values = []

    for Ec in Ec_domain:
        q_inel_lp_min, q_inel_lp_max, _, _ = inelastic(m, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L, q_initial, q_final,
                                                       q_step)
        q_inel_min_lp_values.append(q_inel_lp_min)
        q_inel_max_lp_values.append(q_inel_lp_max)
        Ec_values.append(Ec)

        if q_inel_lp_min >= q_inel_lp_max:
            break

    fs = 50
    plt.figure(figsize=(15, 10))
    plt.plot(q_inel_min_lp_values, Ec_values, linewidth=4, markersize=12)
    plt.plot(q_inel_max_lp_values, Ec_values, linewidth=4, markersize=12)
    plt.title('q vs detuning', fontsize=40)
    plt.xlabel('$q(cm^{-1})$', fontsize=40)
    plt.ylabel('$E_C - E_M$ (meV)', fontsize=40)
    plt.legend(["$q_{min}^{L}$", "$q_{max}^{L}$"], fontsize=40, loc="upper right")
    plt.tick_params(axis='x', labelsize=fs - 20)
    plt.tick_params(axis='y', labelsize=fs - 20)
    plt.show()


# ----------------- Wave-number vs Detuning Phase Diagram ---------------------

def plot_q_vs_detuning(q_initial, q_final, q_step, Ec_initial, Ec_final, Ec_step, m, E0, Lc, Delta, Sigma, Gamma_Ex,
                       Gamma_L, a):
    q_domain = np.arange(q_initial, q_final, q_step)
    Ec_domain = np.arange(Ec_initial, Ec_final, Ec_step)

    q_min_lp_values = []
    q_max_lp_values = []
    Ec_values = []

    for Ec in Ec_domain:
        q_min_lp, q_max_lp, _ = boundary(q_initial, q_final, q_step, m, E0, Ec, Lc, Delta, Sigma, Gamma_Ex, Gamma_L, a)

        q_min_lp_values.append(q_min_lp)
        q_max_lp_values.append(q_max_lp)
        Ec_values.append(Ec)

        if q_min_lp >= q_max_lp:
            break

    fs = 50
    plt.figure(figsize=(15, 10))
    plt.plot(q_min_lp_values, Ec_values, linewidth=4, markersize=12)
    plt.plot(q_max_lp_values, Ec_values, linewidth=4, markersize=12)
    plt.title('q vs detuning', fontsize=40)
    plt.xlabel('$q(cm^{-1})$', fontsize=40)
    plt.ylabel('$E_C - E_M$ (meV)', fontsize=40)
    plt.legend(["$q_{min}^{L}$", "$q_{max}^{L}$"], fontsize=40, loc="upper right")
    plt.tick_params(axis='x', labelsize=fs - 20)
    plt.tick_params(axis='y', labelsize=fs - 20)
    plt.show()

