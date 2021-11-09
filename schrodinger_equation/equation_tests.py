"""
Script created to test the PDEs implemented with DeepXDE library, given the 
solutions of the PDEs.
"""

import numpy as np
import deepxde as dde

from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass
from pinnse.schrodinger_eq_exact_values import TISE_hydrogen_exact

hbar = 1
a0 = 1

def solution_310(input):
    return 4 / (81 * np.sqrt(6)) * (6-input) * input * np.exp(-input/3)
def solution_310_1st_derivative(input):
    return 4 / (81 * np.sqrt(6)) * (6-(4*input)+(input**2 / 3)) * np.exp(-input/3)
def solution_310_2nd_derivative(input):
    return 4 / (81 * np.sqrt(6)) * (-6+(2*input)-(input**2 / 9)) * np.exp(-input/3)

def solution_310_polar(input):
    return np.sqrt(6) * np.cos(input) / 2
def solution_310_1st_derivative_polar(input):
    return -np.sqrt(6) * np.sin(input) / 2
def solution_310_2nd_derivative_polar(input):
    return -np.sqrt(6) * np.cos(input) / 2

def R_pde_np(r, R, dR_dr, dR_drr, n=1, l=0):
    # constants
    electron_mass = (4*pi*epsilon_0) / (e**2) # due to hbar, a0 = 1
    h = hbar * 2 * pi # due to hbar=1
    Z = 1
    A = l * (l+1)
    E_n = - (electron_mass* Z**2 * e**4) / (8 * epsilon_0**2 * h**2 * n**2)
    
    c1 = r**2 * dR_drr
    c2 = 2*r*dR_dr
    # c3 = 2*electron_mass * r**2 * E_n * R_nl / (hbar**2)
    # c4 = electron_mass * Z * e**2 * r * R_nl / (2*hbar**2 * pi * epsilon_0)
    c3 = - r**2 * R / (n**2)
    c4 = 2 * r * R
    c5 = -l*(l+1) * R

    return c1+c2+c3+c4+c5

def f_pde_np(theta, f, df_dtheta, df_dthetatheta, l=1, m=0):

    c1 = (np.sin(theta)**2 * df_dthetatheta) / f
    c2 = np.sin(theta) * np.cos(theta) * df_dtheta / f
    c3 = l*(l+1) * np.sin(theta)**2
    c4 = - m**2

    return c1+c2+c3+c4


def main():
    # geom = dde.geometry.Interval(0,30)
    # X = geom.uniform_points(1000)

    # r = X.reshape(-1)
    # theta = np.linspace(0, pi, 1000)
    # phi = np.linspace(0, 2*pi, 1000)

    # # R_nl_gt, f_lm_gt, g_m_gt = TISE_hydrogen_exact(r, theta, phi, 3, 1, 0)
    # R_nl_gt = solution_310(r)
    # R_nl_gt_1st_derivative = solution_310_1st_derivative(r)
    # R_nl_gt_2nd_derivative = solution_310_2nd_derivative(r)

    # test_results = R_pde_np(r, R_nl_gt, R_nl_gt_1st_derivative, R_nl_gt_2nd_derivative, , n=1, l=0)
    # print(max(test_results))

    ### POLAR TEST ###

    geom = dde.geometry.Interval(0,np.pi)
    X = geom.uniform_points(1000)

    r = np.linspace(0, 30, 1000) # POLAR SETUP
    theta = X.reshape(-1)
    phi = np.linspace(0, 2*pi, 1000)

    # R_nl_gt, f_lm_gt, g_m_gt = TISE_hydrogen_exact(r, theta, phi, 3, 1, 0)
    f_lm_gt = solution_310_polar(theta)
    f_lm_gt_1st_derivative = solution_310_1st_derivative_polar(theta)
    f_lm_gt_2nd_derivative = solution_310_2nd_derivative_polar(theta)

    test_results = f_pde_np(theta, f_lm_gt, f_lm_gt_1st_derivative, f_lm_gt_2nd_derivative, l=1, m=0)
    #print(test_results)
    print(max(test_results))


if __name__ == "__main__":
    main()
