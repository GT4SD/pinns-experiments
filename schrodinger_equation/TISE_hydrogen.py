from deepxde import backend
import os
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import deepxde as dde
from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass, physical_constants
from scipy.integrate import simps
from pinnse.schrodinger_eq_exact_values import TISE_hydrogen_exact

from pinnse.custom_model import CustomLossModel
from pinnse.custom_boundary_conditions import PeriodicBC
from pinnse.custom_pde import CustomPDE
from pinnse.basis_functions import FourierBasis, LegendreBasis

#tf.disable_v2_behavior() # we are using tensorflow.compat.v1

hbar = 1
a0 = 1
#a0 = physical_constants["Bohr radius"][0]

tf.random.set_seed(5)

def weight_condition(inputs):
    # larger the radius, smaller the weights!
    return 1 / inputs
    #return tf.where((inputs>15), tf.constant(1, dtype=inputs.dtype), tf.constant(10, dtype=inputs.dtype))
    #return tf.where((inputs>15), 1 / inputs, tf.constant(1/15, dtype=inputs.dtype))

def cos_prior(args, f): # Prior for prior learning
    x = args[:,0:1]
    y = f[:,0:1]
    return tf.cos(x) + tf.sin(x) + (tf.sin(x) * tf.cos(x)) - y
    #return tf.cos(x) + tf.sin(x) - y
    #return tf.cos(x) - y

def fourier_prior(args, f):
    x = args[:,0:1]
    y = f[:,0:1]
    basis = FourierBasis(max_k=10)
    representation = basis.compute(x)
    return representation - y

def legendre_prior(args, R):
    x = args[:,0:1]
    y = R[:,0:1]
    basis = LegendreBasis(max_n=30)
    representation = basis.compute(x)
    return representation - y

# ---------------------------------
# ---------------------------------
# ---------------------------------

def R_pde(args, R, **quantum_numbers):
    '''
    args[:,0:1] = r
    '''
    n,l,m = quantum_numbers.values()
    
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

def f_pde(args, f, **quantum_numbers):
    n,l,m = quantum_numbers.values()

    theta = args[:,0:1]
    f_lm = f[:,0:1]
    df_dtheta = dde.grad.jacobian(f, args, j=0)
    df_dthetatheta = dde.grad.hessian(f, args, i=0, j=0)

    # c1 = (tf.sin(theta)**2 * df_dthetatheta) / f_lm
    # c2 = tf.sin(theta) * tf.cos(theta) * df_dtheta / f_lm
    # c3 = l*(l+1) * tf.sin(theta)**2
    # c4 = - m**2
    c1 = tf.sin(theta)**2 * df_dthetatheta
    c2 = tf.sin(theta) * tf.cos(theta) * df_dtheta
    c3 = l*(l+1) * (tf.sin(theta)**2) * f_lm
    c4 = - (m**2) * f_lm

    return c1+c2+c3+c4

def g_pde(args, g, **quantum_numbers):
    n,l,m = quantum_numbers.values()

    phi = args[:,0:1]
    g_m = g[:,0:1]
    dg_dphiphi = dde.grad.hessian(g, args, i=0, j=0)

    equation = dg_dphiphi + (m**2) * g_m
    return equation

# ---------------------------------
# ---------------------------------
# ---------------------------------

def learn_R_nl(quantum_numbers, results_path):
    geom = dde.geometry.Interval(0,30) # RADIAL SETUP
    def boundary_right(x, on_boundary):
        return on_boundary and np.isclose(x[0], 30)
    bc = dde.DirichletBC(geom, lambda x:0, boundary_right)
    data = CustomPDE(geom, R_pde, bcs=[bc], num_domain=500, num_boundary=200, pde_extra_arguments=quantum_numbers)

    net = dde.maps.FNN([1] + [32] * 3 + [1], "tanh", "Glorot normal")
    
    model = dde.Model(data, net)
    model.compile("adam", lr=1.0e-5)
    model.train(epochs=20000)
    return model

def learn_f_lm(quantum_numbers, results_path):
    geom = dde.geometry.Interval(0,np.pi) # POLAR SETUP
    def boundary_left(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)
    bc = PeriodicBC(geom, 0, boundary_left, periodicity="antisymmetric")
    data = CustomPDE(geom, f_pde, bcs=[bc], num_domain=500, num_boundary=200, pde_extra_arguments=quantum_numbers)

    net = dde.maps.FNN([1] + [32] * 3 + [1], "tanh", "Glorot normal")

    # model = dde.Model(data, net)
    # model.compile("adam", lr=1.0e-5)
    # model.train(epochs=10000)
    prior_data = dde.data.PDE(geom, cos_prior, bcs=[], num_domain=500)
    prior_save_path = results_path + "f_prior/model"
    compile_train_args = {'optimizer':'adam', 'lr':1e-3, 'epochs':10000}
    model = CustomLossModel(data, net)
    model.learn_prior(prior_data, prior_save_path, **compile_train_args)
    model.compile("adam", lr=1.0e-5, loss='MSE')
    model.train(epochs=20000)
    return model

def learn_g_m(quantum_numbers, results_path):
    geom = dde.geometry.Interval(0,2. * np.pi) # AZIMUTH SETUP
    def boundary_left(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)
    bc = PeriodicBC(geom, 0, boundary_left, periodicity="symmetric")
    data = CustomPDE(geom, g_pde, bcs=[bc], num_domain=500, num_boundary=200, pde_extra_arguments=quantum_numbers)

    net = dde.maps.FNN([1] + [32] * 3 + [1], "tanh", "Glorot normal")

    model = dde.Model(data, net)
    model.compile("adam", lr=1.0e-5)
    model.train(epochs=20000)
    return model

# ---------------------------------
# ---------------------------------
# ---------------------------------

def main():
    results_path = '/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/results_TISE_hydrogen/'
    quantum_numbers = {'n':3, 'l':2, 'm':1}

    # data can be created with the solution function as well!
    # geom = dde.geometry.Interval(0,30) # RADIAL SETUP
    # # bc = dde.DirichletBC(geom, lambda x: 0, 
    # #     lambda x, on_boundary: on_boundary)
    # def boundary_right(x, on_boundary):
    #     return on_boundary and np.isclose(x[0], 30)
    # bc = dde.DirichletBC(geom, lambda x:0, boundary_right)
    # data = CustomPDE(geom, R_pde, bcs=[bc], num_domain=500, num_boundary=200, pde_extra_arguments=quantum_numbers)
    # ---------------------------------
    geom = dde.geometry.Interval(0,np.pi) # POLAR SETUP
    # def boundary_left(x, on_boundary):
    #     return on_boundary and np.isclose(x[0], 0)
    # bc = PeriodicBC(geom, 0, boundary_left, periodicity="antisymmetric")
    data = CustomPDE(geom, f_pde, bcs=[], num_domain=500, num_boundary=200, pde_extra_arguments=quantum_numbers)
    # ---------------------------------
    # geom = dde.geometry.Interval(0,2. * np.pi) # AZIMUTH SETUP
    # def boundary_left(x, on_boundary):
    #     return on_boundary and np.isclose(x[0], 0)
    # bc = PeriodicBC(geom, 0, boundary_left, periodicity="symmetric")
    # #data = dde.data.PDE(geom, g_pde, bcs=[bc], num_domain=500, num_boundary=200)
    # data = CustomPDE(geom, g_pde, bcs=[bc], num_domain=500, num_boundary=200, pde_extra_arguments=quantum_numbers)
    # ---------------------------------

    ## MODEL ##
    net = dde.maps.FNN([1] + [32] * 3 + [1], "tanh", "Glorot normal")
    #net = dde.maps.FNN([1] + [128] * 3 + [64] * 3 + [32] * 3 + [1], "tanh", "Glorot normal")
    # TODO: overcomplicating the network works!
    # ---------------------------------
    model = dde.Model(data, net)
    model.compile("adam", lr=1.0e-3)
    # decay_config = ("inverse time", 5000, 100, True)
    # model.compile("adam", lr=1.0e-5, loss='MSE', decay=decay_config)
    # ---------------------------------
    # model = dde.Model(data, net)
    # decay_config = ("inverse time", 1000, 100)
    # model.compile("adam", lr=1.0e-3, decay=decay_config)
    # ---------------------------------
    # model = CustomLossModel(data, net)
    # model.compile("adam", lr=1.0e-3, weight_condition=weight_condition)
    # --------------------------------- # FOR POLAR
    # prior_data = dde.data.PDE(geom, fourier_prior, bcs=[], num_domain=500, num_boundary=200)
    # prior_save_path = results_path + "f_prior/model"
    # compile_train_args = {'optimizer':'adam', 'lr':1e-3, 'epochs':10000}
    # model = CustomLossModel(data, net)
    # model.learn_prior(prior_data, prior_save_path, **compile_train_args)
    # model.compile("adam", lr=1.0e-3, loss='MSE')
    # --------------------------------- # FOR RADIAL
    # prior_data = dde.data.PDE(geom, legendre_prior, bcs=[], num_domain=500, num_boundary=200)
    # prior_save_path = results_path + "R_prior/model"
    # compile_train_args = {'optimizer':'adam', 'lr':1e-3, 'epochs':10000}
    # model = CustomLossModel(data, net)
    # model.learn_prior(prior_data, prior_save_path, **compile_train_args)
    # decay_config = ("inverse time", 1000, 100)
    # model.compile("adam", lr=1.0e-3, loss='MSE', decay=decay_config)
    # ---------------------------------
    model.train(epochs=20000)
    ############

    #X = geom.random_points(1000)
    X = geom.uniform_points(1000)
    #X = model.train_state.X_train
    #R_nl_pred = model.predict(X)
    f_lm_pred = model.predict(X)
    #g_m_pred = model.predict(X)


    #np.savetxt(results_path+'X.txt', X)
    #np.savetxt(results_path+'R_nl_pred.txt', R_nl_pred)
    #np.savetxt(results_path+'f_lm_pred.txt', f_lm_pred)
    #np.savetxt(results_path+'g_m_pred.txt', g_m_pred)

    ## EVALUATION ##

    # r = X.reshape(-1) # RADIAL SETUP
    # theta = np.linspace(0, pi, 1000)
    # phi = np.linspace(0, 2*pi, 1000)
    r = np.linspace(0, 30, 1000) # POLAR SETUP
    theta = X.reshape(-1)
    phi = np.linspace(0, 2*pi, 1000)
    # r = np.linspace(0, 30, 1000) # AZIMUTH SETUP
    # theta = np.linspace(0, pi, 1000)
    # phi = X.reshape(-1)

    n,l,m = quantum_numbers.values()
    R_nl_gt, f_lm_gt, g_m_gt = TISE_hydrogen_exact(r, theta, phi, n, l, m)

    # ############################
    # ## Radial function test ##
    # ############################
    # R_nl_pred = R_nl_pred.reshape(-1)
    # integrand = R_nl_pred**2 * r**2
    # C = simps(integrand, r)
    # normalization_constant = 1. / np.sqrt(C)

    # plt.figure(figsize=(12,8))
    # #plt.plot(r, 2*np.exp(-r), 'ob')
    # plt.plot(r, R_nl_gt, '.k')
    # plt.plot(r, normalization_constant * R_nl_pred, '.r')
    # #plt.plot(r, r**2 * R_nl**2, '.b') # actually there also is a factor 4pi
    # #plt.show()
    # plt.savefig(results_path + 'results_radial.png')
    # ############################


    ############################
    ## Polar function test ##
    ############################
    f_lm_pred = f_lm_pred.reshape(-1)
    integrand = f_lm_pred**2 * np.sin(theta)
    C = simps(integrand, theta)
    normalization_constant = 1. / np.sqrt(C)
    
    plt.figure(figsize=(12,8))
    #plt.plot(theta, np.sqrt(6/(2*pi))*np.cos(theta)/2, 'ob')
    plt.plot(theta, f_lm_gt, '.k')
    plt.plot(theta, normalization_constant * f_lm_pred, '.r')
    #plt.plot(theta, factor * f_lm_pred, '.b')
    #plt.show()
    plt.savefig(results_path + 'results_polar.png')
    ############################


    # ############################
    # ## Azimuth function test ##
    # ############################
    # g_m_pred = g_m_pred.reshape(-1)
    # integrand = g_m_pred**2
    # C = simps(integrand, phi)
    # normalization_constant = 1. / np.sqrt(C)

    # plt.figure(figsize=(12,8))
    # #plt.plot(theta, np.sqrt(6/(2*pi))*np.cos(theta)/2, 'ob')
    # plt.plot(theta, g_m_gt, '.k')
    # plt.plot(theta, normalization_constant * g_m_pred, '.r')
    # #plt.show()
    # plt.savefig(results_path + 'results_azimuth.png')
    # ############################


def finalize(quantum_numbers = {'n':3, 'l':1, 'm':0}):
    from matplotlib import cm
    results_path = '/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/results_TISE_hydrogen/'
    model_save_path = results_path + 'all_checkpoints/'
    #quantum_numbers = {'n':3, 'l':1, 'm':0}
    n,l,m = quantum_numbers.values()
    number = str(n)+str(l)+str(m)

    model_g = learn_g_m(quantum_numbers, results_path) # results path in case of prior learning
    model_R = learn_R_nl(quantum_numbers, results_path)
    model_f = learn_f_lm(quantum_numbers, results_path)
    # SAVE MODELS #
    # model_g.save(model_save_path+'{number}/g_model/model'.format(number=number))
    # model_f.save(model_save_path+'{number}/f_model/model'.format(number=number))
    # model_R.save(model_save_path+'{number}/R_model/model'.format(number=number))
    # NORMALIZATIONS #
    norm_n = 10000
    r_norm, theta_norm, phi_norm = np.linspace(0, 30, norm_n), np.linspace(0, pi, norm_n), np.linspace(0, 2.*pi, norm_n)
    #plot_gt_R, plot_gt_f, plot_gt_g = TISE_hydrogen_exact(r_norm, theta_norm, phi_norm, 3,1,0)

    r_norm, theta_norm, phi_norm = r_norm.reshape((norm_n, 1)), theta_norm.reshape((norm_n, 1)), phi_norm.reshape((norm_n, 1))
    R_norm = model_R.predict(r_norm)
    R_norm = R_norm.reshape(-1) # normalization
    r_norm = r_norm.reshape(-1)
    integrand = R_norm**2 * r_norm**2
    R_normalization_constant = 1. / np.sqrt( simps(integrand, r_norm) )
    # ------
    f_norm = model_f.predict(theta_norm)
    f_norm = f_norm.reshape(-1) # normalization
    theta_norm = theta_norm.reshape(-1)
    integrand = f_norm**2 * np.sin(theta_norm)
    f_normalization_constant = 1. / np.sqrt( simps(integrand, theta_norm) )
    # ------
    g_norm = model_g.predict(phi_norm)
    g_norm = g_norm.reshape(-1) # normalization
    phi_norm = phi_norm.reshape(-1)
    integrand = g_norm**2
    g_normalization_constant = 1. / np.sqrt( simps(integrand, phi_norm) )

    # ## PLOT FOR TESTING ##
    # fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    # ax1.plot(r_norm, plot_gt_R, '.k')
    # ax1.plot(r_norm, R_normalization_constant * R_norm, '.r')
    # ax2.plot(theta_norm, plot_gt_f, '.k')
    # ax2.plot(theta_norm, f_normalization_constant * f_norm, '.r')
    # ax3.plot(phi_norm, plot_gt_g, '.k')
    # ax3.plot(phi_norm, g_normalization_constant * g_norm, '.r')    
    # plt.savefig(results_path + '310_radial_polar_azimuth.png')

    ## PLOT FOR Y=0 ##
    rmax = 20
    n_points = 1000 # must be an even number
    x = np.linspace(-rmax,rmax,n_points)
    z = np.linspace(-rmax,rmax,n_points)
    X, Z = np.meshgrid(x, z)
    r = np.sqrt(X**2 + Z**2)
    theta = np.arctan(np.sqrt(X**2) / Z)
    theta = np.where(theta<0,np.pi+theta,theta)
    phi = [pi*np.ones([n_points,int(np.floor(n_points/2))])
        , np.zeros([n_points,int(np.ceil(n_points/2))])]
    phi = np.hstack(phi)

    R_nl_gt, f_lm_gt, g_m_gt = TISE_hydrogen_exact(r, theta, phi, n,l,m)
    wavefunction_gt = R_nl_gt*f_lm_gt*g_m_gt
    p_gt = wavefunction_gt**2 # Probability distribution

    r = r.reshape((n_points*n_points, 1))
    R_nl_pred = model_R.predict(r)
    R_nl_pred = R_normalization_constant * R_nl_pred
    R_nl_pred = R_nl_pred.reshape((n_points, n_points))
    
    theta = theta.reshape((n_points*n_points, 1))
    f_lm_pred = model_f.predict(theta)
    f_lm_pred = f_normalization_constant * f_lm_pred
    f_lm_pred = f_lm_pred.reshape((n_points, n_points))

    phi = phi.reshape((n_points*n_points, 1))
    g_m_pred = model_g.predict(phi)
    g_m_pred = g_normalization_constant * g_m_pred
    g_m_pred = g_m_pred.reshape((n_points, n_points))

    wavefunction_pred = R_nl_pred*f_lm_pred*g_m_pred
    p_pred = wavefunction_pred**2

    ## ERROR ANALYSIS
    print("ERROR ANALYSIS:")
    print(np.mean(R_nl_gt - R_nl_pred))
    print(np.mean(f_lm_gt - f_lm_pred))
    print(np.mean(g_m_gt - g_m_pred))
    print("----------------")

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(p_gt,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot)
    ax1.title.set_text('Exact solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')
    ax2.imshow(p_pred,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot)
    ax2.title.set_text('PINN prediction')
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    fig.suptitle('{n},{l},{m} ORBITAL'.format(n=n, l=l, m=m))
    plt.savefig(results_path + "figures/" + '{n}{l}{m}.png'.format(n=n, l=l, m=m))


def finalize_loop():
    keys = ['n', 'l', 'm']
    # orbitals = [(2,0,0), (3,0,0), (2,1,0), (3,1,0), (3,1,1),
    #     (2,1,1), (3,2,0), (3,2,1), (3,2,2), (4,0,0), (4,1,0), 
    #     (4,1,1), (4,2,0), (4,2,1)]
    orbitals = [(2,0,0), (3,0,0), (3,1,0), (3,1,1),
        (2,1,1), (3,2,0), (3,2,2), (4,0,0), 
        (4,1,1), (4,2,0), (4,2,1)]
        # (2,1,0), (4,1,0)
        # (3,2,1) learned with prior learning
    lyman_orbitals = [(1,0,0), (2,1,0), (3,1,0), (4,1,0), (5,1,0) ,(6,1,0)]
    for orbital in lyman_orbitals:
        quantum_numbers = dict(zip(keys, orbital))
        finalize(quantum_numbers)


class MotherSolverTISE():
    """
    Mother solver for hydrogen TISE in unperturbed setting
    """
    def __init__(self, configs):
        """
        config elements:
        - "n_max": maximum n value to consider, orbitals will be estimated up to this n_max value
        - "lr": list of learning rates for seperate networks (radial, polar, azimuth)
        - "epochs": list of epochs for seperate networks (radial, polar, azimuth)
        - "save_path": path to save the results
        """
        self.n_max = configs["n_max"]
        self.lr = configs["lr"]
        self.epochs = configs["epochs"]
        self.save_path = configs["save_path"]
        self.quantum_matrix = None

    def _Rpde(self, args, R, **quantum_matrix):
        quantum_matrix = quantum_matrix["Rmatrix"]
        
        # constants
        electron_mass = (4*pi*epsilon_0) / (e**2) # due to hbar, a0 = 1
        h = hbar * 2 * pi # due to hbar=1
        Z = 1
        #A = l * (l+1)
        #E_n = - (electron_mass* Z**2 * e**4) / (8 * epsilon_0**2 * h**2 * n**2)

        r = args[:,0:1]
        out_list = []
        for index, values in enumerate(quantum_matrix):
            n,l = values
            R_nl = R[:,index:index+1]
            dR_dr = dde.grad.jacobian(R, args, i=index, j=0)
            dR_drr = dde.grad.hessian(R, args, component=index, i=0, j=0)
            
            c1 = r**2 * dR_drr
            c2 = 2*r*dR_dr
            # c3 = 2*electron_mass * r**2 * E_n * R_nl / (hbar**2)
            # c4 = electron_mass * Z * e**2 * r * R_nl / (2*hbar**2 * pi * epsilon_0)
            c3 = - r**2 * R_nl / (n**2)
            c4 = 2 * r * R_nl
            c5 = -l*(l+1) * R_nl
            out_list.append(c1+c2+c3+c4+c5)

        return out_list

    def _fpde(self, args, f, **quantum_matrix):
        quantum_matrix = quantum_matrix["fmatrix"]

        theta = args[:,0:1]
        out_list = []
        for index, values in enumerate(quantum_matrix):
            l,m = values
            f_lm = f[:,index:index+1]
            df_dtheta = dde.grad.jacobian(f, args, i=index, j=0)
            df_dthetatheta = dde.grad.hessian(f, args, component=index, i=0, j=0)

            c1 = tf.sin(theta)**2 * df_dthetatheta
            c2 = tf.sin(theta) * tf.cos(theta) * df_dtheta
            c3 = l*(l+1) * (tf.sin(theta)**2) * f_lm
            c4 = - (m**2) * f_lm
            out_list.append(c1+c2+c3+c4)

        return out_list

    def _gpde(self, args, g, **quantum_matrix):
        quantum_matrix = quantum_matrix["gmatrix"]

        phi = args[:,0:1]
        out_list = []
        for index, values in enumerate(quantum_matrix):
            m = values
            g_m = g[:,index:index+1]
            dg_dphiphi = dde.grad.hessian(g, args, component=index, i=0, j=0)

            equation = dg_dphiphi + (m**2) * g_m
            out_list.append(equation)
        
        return out_list

    def _normalize(self):
        # NORMALIZATIONS #
        norm_n = 10000
        r_norm, theta_norm, phi_norm = np.linspace(0, 30, norm_n), np.linspace(0, pi, norm_n), np.linspace(0, 2.*pi, norm_n)

        model_R, model_f, model_g = self.models

        r_norm, theta_norm, phi_norm = r_norm.reshape((norm_n, 1)), theta_norm.reshape((norm_n, 1)), phi_norm.reshape((norm_n, 1))
        R_norm = model_R.predict(r_norm)
        integrand = R_norm**2 * r_norm**2
        r_norm = r_norm.reshape(-1)
        R_normalization_constants = []
        for i in range(len(self.quantum_matrix["Rmatrix"])):
            R_normalization_constant = 1. / np.sqrt( simps(integrand[:,i], r_norm) )
            R_normalization_constants.append(R_normalization_constant)
        # ------
        f_norm = model_f.predict(theta_norm)
        integrand = f_norm**2 * np.sin(theta_norm)
        theta_norm = theta_norm.reshape(-1)
        f_normalization_constants = []
        for i in range(len(self.quantum_matrix["fmatrix"])):
            f_normalization_constant = 1. / np.sqrt( simps(integrand[:,i], theta_norm) )
            f_normalization_constants.append(f_normalization_constant)
        # ------
        g_norm = model_g.predict(phi_norm)
        integrand = g_norm**2
        phi_norm = phi_norm.reshape(-1)
        g_normalization_constants = []
        for i in range(len(self.quantum_matrix["gmatrix"])):
            g_normalization_constant = 1. / np.sqrt( simps(integrand[:,i], phi_norm) )
            g_normalization_constants.append(g_normalization_constant)

        normalization_constants = [R_normalization_constants, f_normalization_constants, g_normalization_constants]
        return normalization_constants

    def _plot(self, p_gt, p_pred, name, path):
        fig, (ax1, ax2) = plt.subplots(1,2)
        rmax = self.rmax
        ax1.imshow(p_gt,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot)
        ax1.title.set_text('Exact solution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('z')
        ax2.imshow(p_pred,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot)
        ax2.title.set_text('PINN prediction')
        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        fig.suptitle(name)
        plt.savefig(path + name)

    def _solver_setup(self, quantum_matrix):
        ## RADIAL ##
        R_width = np.sum(np.arange(self.n_max + 1)) # width of the network (number of outputs)
        R_width = len(quantum_matrix["Rmatrix"])
        R_geom = dde.geometry.Interval(0,30)
        def R_boundary_right(x, on_boundary):
            return on_boundary and np.isclose(x[0], 30)
        R_bc = [dde.DirichletBC(R_geom, lambda x:0, R_boundary_right, component=i) for i in range(R_width)]
        R_data = CustomPDE(R_geom, self._Rpde, bcs=R_bc, num_domain=1500, num_boundary=600, pde_extra_arguments=quantum_matrix)

        R_backbone = [1] + [64]
        R_neck = [list(np.ones(R_width, dtype=int)*32)] * 3 
        R_head = [R_width]
        R_net = dde.maps.PFNN(R_backbone + R_neck + R_head , "tanh", "Glorot normal")
        
        R_model = dde.Model(R_data, R_net)

        ## POLAR ##
        f_width = len(quantum_matrix["fmatrix"])
        f_geom = dde.geometry.Interval(0,np.pi)
        f_data = CustomPDE(f_geom, self._fpde, bcs=[], num_domain=1500, num_boundary=600, pde_extra_arguments=quantum_matrix)

        f_backbone = [1] + [64]
        f_neck = [list(np.ones(f_width, dtype=int)*32)] * 3 
        f_head = [f_width]
        f_net = dde.maps.PFNN(f_backbone + f_neck + f_head, "tanh", "Glorot normal")

        f_model = dde.Model(f_data, f_net)

        ## AZIMUTH ##
        g_width = len(quantum_matrix["gmatrix"])
        g_geom = dde.geometry.Interval(0,2. * np.pi)
        def g_boundary_left(x, on_boundary):
            return on_boundary and np.isclose(x[0], 0)
        g_bc = [PeriodicBC(g_geom, 0, g_boundary_left, periodicity="symmetric", component=i) for i in range(g_width)]
        g_data = CustomPDE(g_geom, self._gpde, bcs=g_bc, num_domain=1500, num_boundary=600, pde_extra_arguments=quantum_matrix)

        g_backbone = [1] + [64]
        g_neck = [list(np.ones(g_width, dtype=int)*32)] * 3 
        g_head = [g_width]
        g_net = dde.maps.PFNN(g_backbone + g_neck + g_head, "tanh", "Glorot normal")

        g_model = dde.Model(g_data, g_net)

        return [R_model, f_model, g_model]

    def solve(self):
        self._create_quantum_numbers_matrix()
        self.models = self._solver_setup(self.quantum_matrix)
        save_names = ['R_model/model', 'f_model/model', 'g_model/model'] 

        ## SOLVE ##
        models_tmp = []
        for i, model in enumerate(self.models):
            model.compile("adam", lr= self.lr[i])
            model.train(epochs= self.epochs[i])
            model.save(self.save_path + save_names[i])
            models_tmp.append(model)
        self.models = models_tmp

    def evaluate(self, load=False):
        assert ((self.quantum_matrix != None) or load == True), "You have to solve first in order to be able to evaluate!"
        figure_path = self.save_path + "figures/"

        if load:
            self._create_quantum_numbers_matrix()
            self.models = self._solver_setup(self.quantum_matrix)
            models_tmp = []
            save_names = ['R_model/model', 'f_model/model', 'g_model/model'] 
            for i,model in enumerate(self.models):
                model.compile("adam", lr=self.lr[i], loss='MSE')
                pth = os.path.split(self.save_path + save_names[i])[0]
                model.saver.restore(model.sess,tf.train.latest_checkpoint(pth))
                models_tmp.append(model)
            self.models = models_tmp

        model_R, model_f, model_g = self.models
        R_normalization_constant, f_normalization_constant, g_normalization_constant = self._normalize()

        ## PLOT FOR Y=0 ##
        rmax = 20
        self.rmax = rmax
        n_points = 1000 # must be an even number
        x = np.linspace(-rmax,rmax,n_points)
        z = np.linspace(-rmax,rmax,n_points)
        X, Z = np.meshgrid(x, z)
        r = np.sqrt(X**2 + Z**2)
        theta = np.arctan(np.sqrt(X**2) / Z)
        theta = np.where(theta<0,np.pi+theta,theta)
        phi = [pi*np.ones([n_points,int(np.floor(n_points/2))])
            , np.zeros([n_points,int(np.ceil(n_points/2))])]
        phi = np.hstack(phi)

        r_ = r.reshape((n_points*n_points, 1))
        theta_ = theta.reshape((n_points*n_points, 1))
        phi_ = phi.reshape((n_points*n_points, 1))

        R_nl_pred = model_R.predict(r_)
        R_nl_pred = R_normalization_constant * R_nl_pred
        f_lm_pred = model_f.predict(theta_)
        f_lm_pred = f_normalization_constant * f_lm_pred
        g_m_pred = model_g.predict(phi_)
        g_m_pred = g_normalization_constant * g_m_pred

        for i, (n,l1) in enumerate(self.quantum_matrix["Rmatrix"]):
            for j, (l2, m1) in enumerate(self.quantum_matrix["fmatrix"]):
                if l2 == l1:
                    for k, m2 in enumerate(self.quantum_matrix["gmatrix"]): # yes I used 3 for loops, sue me
                        if m1 == m2:
                            n, l1, m1 = int(n), int(l1), int(m1)
                            R_nl_gt, f_lm_gt, g_m_gt = TISE_hydrogen_exact(r, theta, phi, n,l1,m1)
                            wavefunction_gt = R_nl_gt*f_lm_gt*g_m_gt
                            p_gt = wavefunction_gt**2 # Probability distribution

                            R_nl_pred_ = R_nl_pred[:,i:i+1].reshape((n_points, n_points))
                            f_lm_pred_ = f_lm_pred[:,j:j+1].reshape((n_points, n_points))
                            g_m_pred_ = g_m_pred[:,k:k+1].reshape((n_points, n_points))

                            wavefunction_pred = R_nl_pred_*f_lm_pred_*g_m_pred_
                            p_pred = wavefunction_pred**2

                            name = str(n)+str(l1)+str(m1)
                            self._plot(p_gt, p_pred, name=name, path=figure_path)

    def _create_quantum_numbers_matrix(self):
        Rmatrix = []
        for n in range(self.n_max, 0, -1):
            a = np.arange(n)[::-1]
            b = np.ones(n, dtype=int) * n
            ab = [[b[i], a[i]] for i in range(len(a))]
            Rmatrix.extend(ab)
        fmatrix = []
        for l in range(self.n_max-1, -1, -1):
            a = np.arange(-l, l+1)[::-1]
            b = np.ones(2*l +1, dtype=int) * l
            ab = [[b[i], a[i]] for i in range(len(a))]
            fmatrix.extend(ab)
        gmatrix = list(np.arange(-self.n_max+1, self.n_max)[::-1])
        quantum_matrix = {}
        quantum_matrix["Rmatrix"] = Rmatrix
        quantum_matrix["fmatrix"] = fmatrix
        quantum_matrix["gmatrix"] = gmatrix
        self.quantum_matrix = quantum_matrix


def main():
    configs = {
        "n_max": 4,
        "lr": [1e-5, 1e-3, 1e-3],
        "epochs": [20000, 20000, 20000],
        "save_path": "/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/results_TISE_hydrogen/MotherModel/n2/"
    }
    solver = MotherSolverTISE(configs)
    solver.solve()
    solver.evaluate(load=False)


if __name__ == '__main__':
    main()
    #finalize({'n':4, 'l':2, 'm':0})
    #finalize_loop()
