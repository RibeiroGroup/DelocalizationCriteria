from typing import Any
from scipy import optimize
from scipy import special
import numpy as np
from mpmath import *
from mpmath import findroot
from numpy import pi
from scipy.optimize import newton
from sympy import erfc, exp


# ------------------------- Core Functions ---------------------------

def energy_cavity(x, Ec, Lc):
    """Calculate cavity photon dispersion."""
    return Ec * np.sqrt(1 + ((x * Lc) / pi) ** 2)


def energy_lp_zero(x, Ec, E0, Lc, Delta):
    """Calculate the zeroth-order approximation for lower polariton energies.

    :param E0: Mean molecular transition energy
    :param Delta: Half of the Rabi Splitting
    :param Lc: Cavity Length
    :param x: wave-vector number
    :type Ec: Cavity cut-off energy

    """
    result: Any = (E0 + energy_cavity(x, Ec, Lc)) / 2 - np.sqrt(
        Delta ** 2 + ((E0 - energy_cavity(x, Ec, Lc)) / 2) ** 2)
    return result


def energy_up_zero(x, Ec, E0, Lc, Delta):
    """Calculate the zeroth-order approximation for upper polariton energies."""
    result: Any = (E0 + energy_cavity(x, Ec, Lc)) / 2 + np.sqrt(
        Delta ** 2 + ((E0 - energy_cavity(x, Ec, Lc)) / 2) ** 2)
    return result


# ------------------- Electronic and Photonic Weights -----------------------

def c_ex_lp(x, Ec, E0, Lc, Delta):
    """Calculate the total electronic weight in lower polaritonic state."""
    return (Delta ** 2) / (Delta ** 2 + (energy_lp_zero(x, Ec, E0, Lc, Delta) - E0) ** 2)


def c_ex_up(x, Ec, E0, Lc, Delta):
    """Calculate the total electronic weight in upper polaritonic state."""
    return Delta ** 2 / (Delta ** 2 + (energy_up_zero(x, Ec, E0, Lc, Delta) - E0) ** 2)


def c_l_lp(x, Ec, E0, Lc, Delta):
    """Calculate the total photonic weight in lower polaritonic state."""
    return 1 - c_ex_lp(x, Ec, Delta, Lc, E0)


def c_l_up(x, Ec, E0, Lc, Delta):
    """Calculate the total photonic weight in upper polaritonic state."""
    return 1 - c_ex_up(x, Ec, Delta, Lc, E0)


# ----------------------- Group Velocity ---------------------------

def group_velocity_lp(x, Ec, E0, Lc, Delta):
    """Calculate group velocity of lower polariton."""
    return (Ec * (Lc ** 2) * x * c_l_lp(x, Ec, E0, Lc, Delta)) / (pi ** 2)


def group_velocity_up(x, Ec, E0, Lc, Delta):
    """Calculate group velocity of upper polariton."""
    return (Ec * (Lc ** 2) * x * c_l_up(x, Ec, E0, Lc, Delta)) / (pi ** 2)


# ----------------- Inelastic Energy Broadening ---------------------

def gamma_inel_lp(x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L):
    """Calculate inelastic energy broadening of lower polariton."""
    return c_ex_lp(x, Ec, E0, Lc, Delta) * Gamma_Ex + c_l_lp(x, Ec, E0, Lc, Delta) * Gamma_L


def gamma_inel_up(x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L):
    """Calculate inelastic energy broadening of upper polariton."""
    return c_ex_up(x, Ec, E0, Lc, Delta) * Gamma_Ex + c_l_up(x, Ec, E0, Lc, Delta) * Gamma_L


# ----------------- Inelastic Mean Free Path ---------------------

def l_inel_lp(x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L):
    """Calculate inelastic mean free path of lower polariton."""
    return group_velocity_lp(x, Ec, E0, Lc, Delta) / gamma_inel_lp(x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L)


def l_inel_up(x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L):
    """Calculate inelastic mean free path of upper polariton."""
    return group_velocity_up(x, Ec, E0, Lc, Delta) / gamma_inel_up(x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L)


# ----------------- Inelastic Boundary Points ---------------------

def q_inel_lp(x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L):
    return l_inel_lp(x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L) - 1 / x


def q_inel_up(x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L):
    return l_inel_up(x, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L) - 1 / x


# ----------------- Energy Broadening due to Fluctuations ---------------------

def gamma_fluc_lp(x, Ec, E0, Lc, Delta, a):
    """Calculate lower polariton energy broadening due to fluctuations."""
    return ((3 * (pi ** 2)) / 2) * ((a / Lc) ** 3) * ((Delta ** 2) / Ec) * c_ex_lp(x, Ec, E0, Lc, Delta)


def gamma_fluc_up(x, Ec, E0, Lc, Delta, a):
    """Calculate upper polariton energy broadening due to fluctuations."""
    return ((3 * (pi ** 2)) / 2) * ((a / Lc) ** 3) * ((Delta ** 2) / Ec) * c_ex_up(x, Ec, E0, Lc, Delta)


# ----------------- Mean Free Path of Scattering by Fluctuations ---------------------

def l_fluc_lp(x, Ec, E0, Lc, Delta, a):
    """Calculate lower polariton mean free path of scattering by fluctuations."""
    return group_velocity_lp(x, Ec, E0, Lc, Delta) / gamma_fluc_lp(x, Ec, E0, Lc, Delta, a)


def l_fluc_up(x, Ec, E0, Lc, Delta, a):
    """Calculate upper polariton mean free path of scattering by fluctuations."""
    return group_velocity_up(x, Ec, E0, Lc, Delta) / gamma_fluc_up(x, Ec, E0, Lc, Delta, a)


# ----------------- Fluctuations Boundary Points ---------------------

def q_fluc_lp(x, Ec, E0, Lc, Delta, a):
    return l_fluc_lp(x, Ec, E0, Lc, Delta, a) - 1 / x


def q_fluc_up(x, Ec, E0, Lc, Delta, a):
    return l_fluc_up(x, Ec, E0, Lc, Delta, a) - 1 / x


# ----------------- Energy Broadening due to Resonant Scattering ---------------------

# fadeeva function
# exp*erfc
def f(x): return exp(-(x ** 2)) * erfc(-1j*x)

def gamma_res_lp(x, Ec, E0, Lc, Delta, Sigma, imag_part_lp):
    """Calculate lower polariton energy broadening due to resonant scattering."""

    B = ((Delta / Sigma) ** 2) * np.sqrt(np.pi)

    # Prompt the user for the real and imaginary parts of the initial guess
    real_part_lp = energy_lp_zero(x, Ec, E0, Lc, Delta)
    #imag_part_lp = float(input("Enter the imaginary part of the initial guess LP: ") or x*deloc.group_velocity_lp(x,Ec, E0, Lc, Delta))

    # Create the initial guess as a complex number
    initial_guess_lp = complex(real_part_lp, imag_part_lp)
    t = findroot(lambda y: 1j * f((y - E0) / Sigma) * B * Sigma + y - Ec * np.sqrt(1 + ((x * Lc) / np.pi) ** 2),
                 initial_guess_lp)
    return t.imag