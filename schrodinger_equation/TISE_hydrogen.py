from deepxde import backend
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepxde as dde
from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass
from scipy.integrate import simps
from pinnse.schrodinger_eq_exact_values import TISE_hydrogen_exact

from pinnse.custom_model import CustomLossModel
from pinnse.custom_boundary_conditions import PeriodicBC
from pinnse.custom_pde import CustomPDE

hbar = 1
a0 = 1

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
    model.train(epochs=10000)
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
    model.train(epochs=10000)
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
    model.train(epochs=10000)
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
    def boundary_left(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)
    bc = PeriodicBC(geom, 0, boundary_left, periodicity="antisymmetric")
    data = CustomPDE(geom, f_pde, bcs=[bc], num_domain=500, num_boundary=200, pde_extra_arguments=quantum_numbers)
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
    # ---------------------------------
    # model = dde.Model(data, net)
    # model.compile("adam", lr=1.0e-5)
    # ---------------------------------
    # model = dde.Model(data, net)
    # decay_config = ("inverse time", 1000, 100)
    # model.compile("adam", lr=1.0e-3, decay=decay_config)
    # ---------------------------------
    # model = CustomLossModel(data, net)
    # model.compile("adam", lr=1.0e-3, weight_condition=weight_condition)
    # ---------------------------------
    prior_data = dde.data.PDE(geom, cos_prior, bcs=[], num_domain=500)
    prior_save_path = results_path + "f_prior/model"
    compile_train_args = {'optimizer':'adam', 'lr':1e-3, 'epochs':10000}
    model = CustomLossModel(data, net)
    model.learn_prior(prior_data, prior_save_path, **compile_train_args)
    model.compile("adam", lr=1.0e-5, loss='MSE')
    # ---------------------------------
    model.train(epochs=10000)
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
    #quantum_numbers = {'n':3, 'l':1, 'm':0}

    model_g = learn_g_m(quantum_numbers, results_path) # results path in case of prior learning
    model_R = learn_R_nl(quantum_numbers, results_path)
    model_f = learn_f_lm(quantum_numbers, results_path)
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

    n,l,m = quantum_numbers.values()
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

    # ## ERROR ANALYSIS
    # print("ERROR ANALYSIS:")
    # print(np.mean(R_nl_gt - R_nl_pred))
    # print(np.mean(f_lm_gt - f_lm_pred))
    # print(np.mean(g_m_gt - g_m_pred))
    # print("----------------")

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
    plt.savefig(results_path + '{n}{l}{m}.png'.format(n=n, l=l, m=m))


def finalize_loop():
    keys = ['n', 'l', 'm']
    # orbitals = [(2,0,0), (3,0,0), (2,1,0), (3,1,0), (3,1,1),
    #     (2,1,1), (3,2,0), (3,2,1), (3,2,2), (4,0,0), (4,1,0), 
    #     (4,1,1), (4,2,0), (4,2,1)]
    orbitals = [(2,0,0), (3,0,0), (3,1,0), (3,1,1),
        (2,1,1), (3,2,0), (3,2,1), (3,2,2), (4,0,0), 
        (4,1,1), (4,2,0), (4,2,1)]
        # (2,1,0), (4,1,0)
        # (3,2,1) learned with prior learning
    for orbital in orbitals:
        quantum_numbers = dict(zip(keys, orbital))
        finalize(quantum_numbers)



if __name__ == '__main__':
    #main()
    finalize({'n':3, 'l':2, 'm':1})
    #finalize_loop()

# azimuth: 6.79e-09
# polar: 5.58e-08
# radial: 6.89e-04