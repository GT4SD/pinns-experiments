import numpy as np
from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass, physical_constants

a0 = physical_constants["Bohr radius"][0]
from scipy.special import genlaguerre, lpmv, jv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from .utils import *


def TISE_hydrogen_exact(r, theta, phi, n=1, l=0, m=0, Z=1):
    """
    https://www.bcp.fu-berlin.de/en/chemie/chemie/forschung/PhysTheoChem/agkeller/_Docs/TC_WS15_Ex07.pdf
    - Give r, theta, phi as np.arrays, they must have the same shape. They represent the spherical values
        on the given grid (array).
    - n, l, m are the quantum numbers.
    - Only returns the real part of the orbitals.
    """
    R_prefactor = np.sqrt(
        ((2 * Z) / (a0 * n)) ** 3
        * np.math.factorial(n - l - 1)
        / (2 * n * np.math.factorial(n + l))
    )
    R_nl = (
        R_prefactor
        * np.exp(-r * Z / (n * a0))
        * (2 * r * Z / (n * a0)) ** l
        * genlaguerre(n - l - 1, 2 * l + 1)(2 * r * Z / (n * a0))
    )

    f_prefactor = (-1) ** m * np.sqrt(
        (2 * l + 1) * np.math.factorial(l - m) / (4 * pi * np.math.factorial(l + m))
    )
    # f_prefactor = np.sqrt( (2*l+1)*np.math.factorial(l-m) / (4*pi*np.math.factorial(l+m)) )
    f_lm = f_prefactor * lpmv(m, l, np.cos(theta))
    f_lm = f_lm * np.sqrt(2 * pi)  # for normalization

    g_m_real = np.cos(m * phi) / np.sqrt(2 * pi)
    g_m_imaginary = np.sin(m * phi) / np.sqrt(2 * pi)
    # g_m = np.vstack([g_m_real, g_m_imaginary])

    return R_nl, f_lm, g_m_real


def TISE_stark_effect_exact(r, theta, phi, electric_field, n=2, m=0):
    """
    http://www.physics.drexel.edu/~bob/Manuscripts/stark.pdf
    - n, m is the manifold to consider (So we have non-degenerate energy values!)
    - returns: eigenvalues, eigenvectors acting on r, theta, phi
    """
    assert n > m, "n must have a greater value than m!"
    factor = -3 * e * electric_field * a0 / 2
    m_abs = np.abs(m)
    if (n - m_abs) > 1:
        l_values = [x for x in range(m_abs + 1, n)]
        M = np.zeros((len(l_values) + 1, len(l_values) + 1))  # initialize matrix
        for i, l in enumerate(l_values):
            R = n * np.sqrt((n - l) * (n + l))
            A = np.sqrt((l + m_abs) * (l - m_abs) / ((2 * l + 1) * (2 * l - 1)))
            M[i][i + 1] = A * R
            M[i + 1][i] = A * R
        w, v = np.linalg.eig(M)  # Eigenvalues / Eigenvectors
        l_values = [x for x in range(m_abs, n)]
        basis_list = []
        for l in l_values:
            basis_list.append(TISE_hydrogen_exact(r, theta, phi, n=n, l=l, m=m))
        basis_list = [R * f * g for (R, f, g) in basis_list]
        basis_array = np.stack(basis_list, axis=-1)
        output_list = []
        for i in range(len(w)):
            val = np.matmul(basis_array, v[:, i])
            output_list.append(np.expand_dims(val, axis=0))
        eigenvectors = np.vstack(output_list)
        eigenvalues = factor * w
    else:
        eigenvalues = np.array([0])
        R, f, g = TISE_hydrogen_exact(r, theta, phi, n=n, l=m_abs, m=m)
        eigenvector = R * f * g
        eigenvectors = np.expand_dims(eigenvector, axis=0)
    return eigenvalues, eigenvectors


def TDSE_unperturbed_exact(r, theta, phi, t, n=1, l=0, m=0, Z=1):
    statitonarys = TISE_hydrogen_exact(r, theta, phi, n=n, l=l, m=m, Z=Z)
    E_n = -(electron_mass * e**4) / (8 * epsilon_0**2 * h**2 * n**2)

    real = statitonarys[0] * statitonarys[1] * statitonarys[2] * np.cos(E_n * t / hbar)
    imag = -statitonarys[0] * statitonarys[1] * statitonarys[2] * np.sin(E_n * t / hbar)

    return real, imag


def TDSE_one_level_AC_exact(
    t,
    field_amplitude,
    omega,
    dipole_moment,
    polarizability,
    k_max=1000,
    s_max=1000,
    backend="numpy",
):
    assert backend in [
        "numpy",
        "tensorflow",
    ], "Only numpy, tensorflow backends supported!"

    quasi_energy = -0.25 * polarizability * field_amplitude**2

    population_amplitudes = []
    for k in range(-k_max, k_max + 1):
        amplitude = 0
        for s in range(-s_max, s_max + 1):
            factor1 = jv(
                s, polarizability * field_amplitude**2 / (8 * omega)
            )  # Bessel function
            factor2 = jv(k + 2 * s, dipole_moment * field_amplitude / omega)
            amplitude += factor1 * factor2
        amplitude *= (-1) ** k
        population_amplitudes.append(amplitude)

    wf_real, wf_imaginary = floquet_theorem(
        t, quasi_energy, population_amplitudes, omega, k_max=k_max, backend=backend
    )

    return wf_real, wf_imaginary


def TISE_hydrogen_1d_exact(x, n=1):
    """
    http://theorie2.physik.uni-greifswald.de/member/bronold/SS2009/1DHatom.pdf
    functions 20a, 20b
    """
    prefactor = np.sqrt(4 / (a0**2 * n**3))
    phi_n_plus = (
        prefactor
        * (-1) ** (n - 1)
        * x
        * genlaguerre(n - 1, 1)(2 * x / (n * a0))
        * np.exp(-x / (n * a0))
    )  # defined for x>0
    phi_n_minus = (
        prefactor
        * (-1) ** (n)
        * x
        * genlaguerre(n - 1, 1)(2 * x / (n * a0))
        * np.exp(x / (n * a0))
    )  # defined for x<0
    return phi_n_plus, phi_n_minus


def TISE_hydrogen_1d_momentum_space_exact(p, n=1):
    """
    http://theorie2.physik.uni-greifswald.de/member/bronold/SS2009/1DHatom.pdf
    functions 16a, 16b
    These functions are defined up to n=3
    """
    assert n in [1, 2, 3], "Exact solution defined for only n=1,2,3"
    prefactor = np.sqrt(2 * a0 * n / (pi * hbar))
    same_factor = 1.0 / (1 + (n * p * a0 / hbar) ** 2)
    alpha = n * p * a0 / hbar
    u, v = (1 - alpha**2) / (1 + alpha**2), (2 * alpha) / (1 + alpha**2)
    beta, gamma = (u**2 - v**2), 2 * u * v
    if n == 1:
        phi_n_plus_real = prefactor * same_factor * u
        phi_n_plus_imaginary = prefactor * same_factor * (-v)
        phi_n_minus_real = prefactor * same_factor * u
        phi_n_minus_imaginary = prefactor * same_factor * (v)
    if n == 2:
        phi_n_plus_real = prefactor * same_factor * beta
        phi_n_plus_imaginary = prefactor * same_factor * (-gamma)
        phi_n_minus_real = prefactor * same_factor * beta
        phi_n_minus_imaginary = prefactor * same_factor * (gamma)
    if n == 3:  # TODO: implement this as recursive method for n>3
        beta, gamma = (beta**2 - gamma**2), 2 * beta * gamma
        phi_n_plus_real = prefactor * same_factor * beta
        phi_n_plus_imaginary = prefactor * same_factor * (-gamma)
        phi_n_minus_real = prefactor * same_factor * beta
        phi_n_minus_imaginary = prefactor * same_factor * (gamma)
    return [
        [phi_n_plus_real, phi_n_plus_imaginary],
        [phi_n_minus_real, phi_n_minus_imaginary],
    ]
