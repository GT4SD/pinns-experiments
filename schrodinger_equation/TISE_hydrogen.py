import yaml
import argparse

from deepxde import backend
import os
import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import deepxde as dde
from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass, physical_constants
a0 = physical_constants["Bohr radius"][0]
from scipy.integrate import simps
from pinnse.schrodinger_eq_exact_values import TISE_hydrogen_exact

from pinnse.custom_model import CustomLossModel
from pinnse.custom_boundary_conditions import PeriodicBC
from pinnse.custom_pde import CustomPDE
from pinnse.basis_functions import FourierBasis, LegendreBasis, LaguerreBasis

#tf.disable_v2_behavior() # we are using tensorflow.compat.v1

# ATOMIC UNITS
hbar = 1
a0 = 1
e = 1
electron_mass = 1
epsilon_0 = 1 / (4*pi)
h = hbar * 2 * pi

tf.random.set_seed(5)

def plot(p_gt, p_pred, rmax, results_path, quantum_numbers):
    n,l,m = quantum_numbers.values()
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,15))
    plt1 = ax1.imshow(p_gt,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot)
    ax1.title.set_text('Exact solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')
    fig.colorbar(plt1,ax=ax1,fraction=0.046, pad=0.04)
    plt2 = ax2.imshow(p_pred,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot)
    ax2.title.set_text('PINN prediction')
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    fig.colorbar(plt2,ax=ax2,fraction=0.046, pad=0.04)
    plt3 = ax3.imshow(np.abs(p_gt-p_pred),extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot)
    ax3.title.set_text('Absolute error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')
    fig.colorbar(plt3,ax=ax3,fraction=0.046, pad=0.04)
    fig.suptitle('{n},{l},{m} ORBITAL'.format(n=n, l=l, m=m))
    plt.savefig(results_path + '{n}{l}{m}.png'.format(n=n, l=l, m=m))

def fourier_prior(args, f):
    x = args[:,0:1]
    y = f[:,0:1]
    basis = FourierBasis(max_k=10)
    representation = basis.compute(x)
    return representation - y

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

    x = args[:,0:1] # phi 
    u, v = g[:,0:1], g[:,1:2] # real/imaginary parts

    du_dx = dde.grad.jacobian(g, args, i=0, j=0)
    dv_dx = dde.grad.jacobian(g, args, i=1, j=0)

    du_dxx = dde.grad.hessian(g, args, i=0, j=0, component=0)
    dv_dxx = dde.grad.hessian(g, args, i=0, j=0, component=1)

    equation_real = du_dxx + (m**2) * u
    equation_imag = dv_dxx + (m**2) * v

    eq_m1 = du_dx + m*v
    eq_m2 = dv_dx - m*u
    #return [equation_real, equation_imag]
    return [equation_real, equation_imag, eq_m1, eq_m2]

# ===============================

def create_model_R_nl(experiment_params=None, network_params=None):
    """
    Given network_params are: network_params["R"]
    """
    assert experiment_params is not None and network_params is not None, "No parameters given!"

    quantum_numbers = experiment_params["quantum_numbers"]
    radial_extent = experiment_params["radial_extent"]

    geom = dde.geometry.Interval(0,radial_extent) # RADIAL SETUP
    def boundary_right(x, on_boundary):
        return on_boundary and np.isclose(x[0], radial_extent)
    bc = dde.DirichletBC(geom, lambda x:0, boundary_right)
    data = CustomPDE(geom, R_pde, bcs=[bc], num_domain=network_params["num_domain"],\
         num_boundary=network_params["num_boundary"], pde_extra_arguments=quantum_numbers)

    if network_params["backbone"] == "FNN":
        net = dde.maps.FNN(network_params["layers"], "tanh", "Glorot normal")
    else:
        raise NotImplementedError("Only FNN backbone is experiment ready!")
    
    model = dde.Model(data, net)
    #model.compile("adam", lr=1.0e-3)
    return model, geom, data, net

def create_model_f_lm(experiment_params=None, network_params=None, results_path=None):
    """
    Given network_params are: network_params["f"]
    """
    assert experiment_params is not None and network_params is not None and\
        results_path is not None, "No parameters given!"

    quantum_numbers = experiment_params["quantum_numbers"]

    geom = dde.geometry.Interval(0,np.pi) # POLAR SETUP
    data = CustomPDE(geom, f_pde, bcs=[], num_domain=network_params["num_domain"],\
         num_boundary=network_params["num_boundary"], pde_extra_arguments=quantum_numbers)

    if network_params["backbone"] == "FNN":
        net = dde.maps.FNN(network_params["layers"], "tanh", "Glorot normal")
    else:
        raise NotImplementedError("Only FNN backbone is experiment ready!")

    if network_params["prior"] is None:
        model = dde.Model(data, net)
        return model, geom, data, net
    elif network_params["prior"] == "Fourier":
        prior_data = dde.data.PDE(geom, fourier_prior, bcs=[], num_domain=500, num_boundary=200)
        prior_save_path = results_path + "f_prior/model"
        compile_train_args = {'optimizer':'adam', 'lr':1e-3, 'epochs':10000}
        model = CustomLossModel(data, net)
        #model.learn_prior(prior_data, prior_save_path, **compile_train_args)

    #model.compile("adam", lr=1.0e-3)
    return model, geom, data, net, prior_data, prior_save_path, compile_train_args

def create_model_g_m(experiment_params=None, network_params=None):
    """
    Given network_params are: network_params["f"]
    """
    assert experiment_params is not None and network_params is not None, "No parameters given!"

    quantum_numbers = experiment_params["quantum_numbers"]

    geom = dde.geometry.Interval(0,2. * np.pi) # AZIMUTH SETUP
    # def boundary_left(x, on_boundary): # for mathematical completeness
    #     return on_boundary and np.isclose(x[0], 0)
    # bc = PeriodicBC(geom, 0, boundary_left, periodicity="symmetric")
    # data = CustomPDE(geom, g_pde, bcs=[bc], num_domain=500, num_boundary=200, pde_extra_arguments=quantum_numbers)
    def u_func(x):
        phi = x[:,0:1]
        n, l, m = quantum_numbers.values()
        R, f, g = TISE_hydrogen_exact(0, 0, phi, n,l,m)
        return g
    def v_func(x):
        phi = x[:,0:1]
        n, l, m = quantum_numbers.values()
        g = np.sin(m*phi) / np.sqrt(2*pi)
        return g
    bc_strict_u = dde.DirichletBC(geom, u_func, lambda _, on_boundary: on_boundary, component=0)
    bc_strict_v = dde.DirichletBC(geom, v_func, lambda _, on_boundary: on_boundary, component=1)
    data = CustomPDE(geom, g_pde, bcs=[bc_strict_u, bc_strict_v], num_domain=network_params["num_domain"],\
         num_boundary=network_params["num_boundary"], pde_extra_arguments=quantum_numbers)

    if network_params["backbone"] == "FNN":
        net = dde.maps.FNN(network_params["layers"], "tanh", "Glorot normal")
    else:
        raise NotImplementedError("Only FNN backbone is experiment ready!")

    model = dde.Model(data, net)
    #model.compile("adam", lr=1.0e-3)
    return model, geom, data, net

def main(config_path=None):

    if config_path is None:
        root = "/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/"
        config_path = root + "experiment_configs/TISE_hydrogen_without_decomposition/config_210.yaml"
    
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    experiment_params = config["experiment_params"]
    network_params = config["network_params"]
    results_path = config["results_path"]
    #results_path = '/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/results_TISE_hydrogen/' # path in local

    quantum_numbers = experiment_params["quantum_numbers"]

    folder = str(quantum_numbers["n"]) + str(quantum_numbers["l"]) + str(quantum_numbers["m"]) + '/'
    results_path = results_path + folder
    import os
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # PLOTTING SAMPLING FOR Y=0
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

    r_model = r.reshape((n_points*n_points, 1))
    theta_model = theta.reshape((n_points*n_points, 1))
    phi_model = phi.reshape((n_points*n_points, 1))

    # =============== #
    # === R train === #
    # =============== #
    R_params = network_params["R"]
    model, geom, data, net = create_model_R_nl(experiment_params, R_params)
    assert len(R_params["lrs"]) == len(R_params["optimizers"]) and \
        len(R_params["lrs"]) == len(R_params["epochs"]), "Incompatible network parameter lengths!"
    for i in range(len(R_params["lrs"])):
        lr = float(R_params["lrs"][i])
        optimizer = R_params["optimizers"][i]
        epoch = R_params["epochs"][i]
        model.compile(optimizer=optimizer, lr=lr)
        # with open(results_path+"model/best_step.txt") as f: # TODO: add this after first loop
        #     best = f.readlines()[0]
        # model.restore(results_path + "model/model.ckpt-" + best, verbose=1)
        checker = dde.callbacks.ModelCheckpoint(
            results_path+"model_R/model.ckpt", save_better_only=True, period=1000
        )
        losshistory, train_state = model.train(epochs=epoch, callbacks=[checker])
        with open(results_path+"model_R/best_step.txt", "w") as text_file:
            text_file.write(str(train_state.best_step))

    # NORMALIZATION
    X = geom.uniform_points(1000)
    r_norm = X.reshape(-1)
    R_nl_pred = model.predict(X)
    R_nl_pred = R_nl_pred.reshape(-1)
    integrand = R_nl_pred**2 * r_norm**2
    C = simps(integrand, r_norm)
    normalization_constant_R = 1. / np.sqrt(C)

    # PLOTTING VALUES
    R_nl_plot = model.predict(r_model)
    R_nl_plot = normalization_constant_R * R_nl_plot

    tf.compat.v1.reset_default_graph()


    # =============== #
    # === f train === #
    # =============== #
    f_params = network_params["f"]
    assert len(f_params["lrs"]) == len(f_params["optimizers"]) and \
        len(f_params["lrs"]) == len(f_params["epochs"]), "Incompatible network parameter lengths!"
    if f_params["prior"] == "Fourier":
        model, geom, data, net, prior_data, prior_save_path, compile_train_args =\
             create_model_f_lm(experiment_params, f_params, results_path)
        model.learn_prior(prior_data, prior_save_path, **compile_train_args)  
    else: 
        model, geom, data, net = create_model_f_lm(experiment_params, f_params, results_path)
    for i in range(len(f_params["lrs"])):
        lr = float(f_params["lrs"][i])
        optimizer = f_params["optimizers"][i]
        epoch = f_params["epochs"][i]
        model.compile(optimizer=optimizer, lr=lr, loss="MSE")
        # with open(results_path+"model/best_step.txt") as f: # TODO: add this after first loop
        #     best = f.readlines()[0]
        # model.restore(results_path + "model/model.ckpt-" + best, verbose=1)
        checker = dde.callbacks.ModelCheckpoint(
            results_path+"model_f/model.ckpt", save_better_only=True, period=1000
        )
        losshistory, train_state = model.train(epochs=epoch, callbacks=[checker])
        with open(results_path+"model_f/best_step.txt", "w") as text_file:
            text_file.write(str(train_state.best_step))

    # NORMALIZATION
    X = geom.uniform_points(1000)
    theta_norm = X.reshape(-1)
    f_lm_pred = model.predict(X)
    f_lm_pred = f_lm_pred.reshape(-1)
    integrand = f_lm_pred**2 * np.sin(theta_norm)
    C = simps(integrand, theta_norm)
    normalization_constant_f = 1. / np.sqrt(C)

    # PLOTTING VALUES
    f_lm_plot = model.predict(theta_model)
    f_lm_plot = normalization_constant_f * f_lm_plot

    tf.compat.v1.reset_default_graph()


    # =============== #
    # === g train === #
    # =============== #
    g_params = network_params["g"]
    model, geom, data, net = create_model_g_m(experiment_params, g_params)
    assert len(g_params["lrs"]) == len(g_params["optimizers"]) and \
        len(g_params["lrs"]) == len(g_params["epochs"]), "Incompatible network parameter lengths!"
    for i in range(len(g_params["lrs"])):
        lr = float(g_params["lrs"][i])
        optimizer = g_params["optimizers"][i]
        epoch = g_params["epochs"][i]
        model.compile(optimizer=optimizer, lr=lr)
        # with open(results_path+"model/best_step.txt") as f: # TODO: add this after first loop
        #     best = f.readlines()[0]
        # model.restore(results_path + "model/model.ckpt-" + best, verbose=1)
        checker = dde.callbacks.ModelCheckpoint(
            results_path+"model_g/model.ckpt", save_better_only=True, period=1000
        )
        losshistory, train_state = model.train(epochs=epoch, callbacks=[checker])
        with open(results_path+"model_g/best_step.txt", "w") as text_file:
            text_file.write(str(train_state.best_step))

    # NORMALIZATION
    X = geom.uniform_points(1000)
    phi_norm = X.reshape(-1)
    g_m_pred = model.predict(X)
    g_m_pred_real, g_m_pred_imag = g_m_pred[:,0], g_m_pred[:,1]
    integrand = g_m_pred_real**2 +  g_m_pred_imag**2
    C = simps(integrand, phi_norm)
    normalization_constant_g = 1. / np.sqrt(C)

    # PLOTTING VALUES
    g_m_plot = model.predict(phi_model)[:,0:1] # consider real values for plotting
    g_m_plot = normalization_constant_g * g_m_plot

    tf.compat.v1.reset_default_graph()


    ## PLOT FOR Y=0 ##
    n,l,m = quantum_numbers.values()
    R_nl_gt, f_lm_gt, g_m_gt = TISE_hydrogen_exact(r, theta, phi, n,l,m)
    wavefunction_gt = R_nl_gt*f_lm_gt*g_m_gt # we can consider only real part since we are plotting for Y=0
    p_gt = wavefunction_gt**2 # Probability distribution

    wavefunction_pred = R_nl_plot*f_lm_plot*g_m_plot
    wavefunction_pred = wavefunction_pred.reshape((n_points, n_points))
    p_pred = wavefunction_pred**2

    plot(p_gt, p_pred, rmax, results_path, quantum_numbers)


if __name__ == "__main__":
    
    root = "/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/"
    #root = "./"
    default_config_path = root + "experiment_configs/TISE_hydrogen/config_210.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, \
        default=default_config_path ,help='Path to the experiment config file.')
    args = parser.parse_args()
    config_path = args.config_path

    main(config_path)

