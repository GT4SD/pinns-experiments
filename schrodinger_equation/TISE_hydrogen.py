import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepxde as dde
from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass
from pinnse.schrodinger_eq_exact_values import TISE_hydrogen_exact

from pinnse.custom_model import CustomLossModel

hbar = 1
a0 = 1

tf.random.set_seed(5)

def weight_condition(inputs):
    # larger the radius, smaller the weights!
    return 1 / inputs
    #return tf.where((inputs>15), tf.constant(1, dtype=inputs.dtype), tf.constant(10, dtype=inputs.dtype))
    #return tf.where((inputs>15), 1 / inputs, tf.constant(1/15, dtype=inputs.dtype))


# def solution_310(input):
#     return 4 / (81 * np.sqrt(6)) * (6-input) * input * np.exp(-input/3)


# let's first test only for R
def R_pde(args, R):
    '''
    args[:,0:1] = r
    '''
    n,l,m = 3,1,0 # TODO: make n,l,m inputs
    
    # constants
    electron_mass = (4*pi*epsilon_0) / (e**2) # due to hbar, a0 = 1
    h = hbar * 2 * pi # due to hbar=1
    Z = 1
    A = l * (l+1)
    E_n = - (electron_mass* Z**2 * e**4) / (8 * epsilon_0**2 * h**2 * n**2)

    r = args[:,0:1]
    R_nl = R[:,0:1]
    dR_dr = dde.grad.jacobian(R, args, j=0)
    dR_drr = dde.grad.hessian(R, args, i=0, j=0)
    
    c1 = r**2 * dR_drr
    c2 = 2*r*dR_dr
    # c3 = 2*electron_mass * r**2 * E_n * R_nl / (hbar**2)
    # c4 = electron_mass * Z * e**2 * r * R_nl / (2*hbar**2 * pi * epsilon_0)
    c3 = - r**2 * R_nl / (n**2)
    c4 = 2 * r * R_nl
    c5 = -l*(l+1) * R_nl

    return c1+c2+c3+c4+c5


def f_pde(args, f):
    n,l,m = 3,1,0

    theta = args[:,0:1]
    f_lm = f[:,0:1]
    df_dtheta = dde.grad.jacobian(f, args, j=0)
    df_dthetatheta = dde.grad.hessian(f, args, i=0, j=0)

    c1 = (tf.sin(theta)**2 * df_dthetatheta) / f_lm
    c2 = tf.sin(theta) * tf.cos(theta) * df_dtheta / f_lm
    c3 = l*(l+1) * tf.sin(theta)**2
    c4 = - m**2

    return c1+c2+c3+c4


def main():
    results_path = '/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/results_TISE_hydrogen/'

    # data can be created with the solution function as well!
    geom = dde.geometry.Interval(0,30)
    data = dde.data.PDE(geom, R_pde, bcs=[], num_domain=500)
    # geom = dde.geometry.Interval(0,np.pi)
    # data = dde.data.PDE(geom, f_pde, bcs=[], num_domain=500)
    X = geom.uniform_points(1000)

    ## MODEL ##
    net = dde.maps.FNN([1] + [32] * 3 + [1], "tanh", "Glorot normal")
    # model = dde.Model(data, net)
    # model.compile("adam", lr=1.0e-3)
    model = CustomLossModel(data, net)
    model.compile("adam", lr=1.0e-3, weight_condition=weight_condition)
    model.train(epochs=10000)
    ############

    #X = geom.random_points(1000)
    X = geom.uniform_points(1000)
    #X = model.train_state.X_train
    R_nl_pred = model.predict(X)
    #f_lm_pred = model.predict(X)


    np.savetxt(results_path+'X.txt', X)
    np.savetxt(results_path+'R_nl_pred.txt', R_nl_pred)
    #np.savetxt(results_path+'f_lm_pred.txt', f_lm_pred)

    ## EVALUATION ##

    r = X.reshape(-1)
    theta = np.linspace(0, pi, 1000)
    phi = np.linspace(0, 2*pi, 1000)
    # r = np.linspace(0, 30, 1000)
    # theta = X.reshape(-1)
    # phi = np.linspace(0, 2*pi, 1000)

    R_nl_gt, f_lm_gt, g_m_gt = TISE_hydrogen_exact(r, theta, phi, 3, 1, 0)

    ############################
    ## Radial function test ##
    ############################
    plt.figure(figsize=(12,8))
    #plt.plot(r, 2*np.exp(-r), 'ob')
    plt.plot(r, R_nl_gt, '.k')
    plt.plot(r, R_nl_pred, '.r')
    #plt.plot(r, r**2 * R_nl**2, '.b') # actually there also is a factor 4pi
    #plt.show()
    ############################

    # ############################
    # ## Polar function test ##
    # ############################
    # plt.figure(figsize=(12,8))
    # plt.plot(theta, np.sqrt(6/(2*pi))*np.cos(theta)/2, 'ob')
    # plt.plot(theta, f_lm_gt, '.k')
    # plt.plot(theta, f_lm_pred, '.r')
    # #plt.show()
    # ############################

    plt.savefig(results_path + 'results.png')



if __name__ == '__main__':
    main()
