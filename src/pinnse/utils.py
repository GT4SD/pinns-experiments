import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass, physical_constants
a0 = physical_constants['Bohr radius'][0]
from scipy.special import genlaguerre, lpmv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pinnse.integrate import integrate_1d

# ATOMIC UNITS
hbar = 1
a0 = 1
e = 1
electron_mass = 1
epsilon_0 = 1 / (4*pi)
h = hbar * 2 * pi


def floquet_theorem(t, quasi_energy, population_amplitudes, omega, k_max=1000, backend="numpy"):
    """
    Wavefunction (or qm-amplitude in interaction picture) 
    - population_amplitudes: list of len 2*k_max + 1 --> amplitude component for each k
    """
    assert backend in ["numpy", "tensorflow"], "Only numpy, tensorflow backends supported!"
    # TODO: change above explanation after checking the mathematics
    fourier_series_real = 0
    fourier_series_imaginary = 0
    for k in range(-k_max, k_max+1):
        i = k + k_max
        period = quasi_energy + k*omega
        amplitude = population_amplitudes[i]
        if backend == "numpy":
            fourier_series_real += amplitude * np.cos(period * t)
            fourier_series_imaginary += - (amplitude * np.sin(period * t)) # omit the minus sign to add later
        elif backend == "tensorflow":
            fourier_series_real += amplitude * tf.cos(period * t)
            fourier_series_imaginary += - (amplitude * tf.sin(period * t)) # omit the minus sign to add later

    return fourier_series_real, fourier_series_imaginary

def count_zero_in_decimal_number(number):
    zeros = 0
    while number < 1:
        number *= 10
        zeros += 1
    return zeros