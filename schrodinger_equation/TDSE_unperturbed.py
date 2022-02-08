import yaml
import argparse

from deepxde import backend
from deepxde.geometry import timedomain
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepxde as dde
from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass, physical_constants
a0 = physical_constants["Bohr radius"][0]
from scipy.integrate import simps
from pinnse.schrodinger_eq_exact_values import TISE_hydrogen_exact, TISE_stark_effect_exact
from matplotlib import cm

from pinnse.custom_model import CustomLossModel
from pinnse.custom_boundary_conditions import PeriodicBC
from pinnse.custom_pde import CustomPDE, CustomTimePDE
from pinnse.basis_functions import FourierBasis

# ATOMIC UNITS
hbar = 1
a0 = 1
e = 1
electron_mass = 1
epsilon_0 = 1 / (4*pi)
h = hbar * 2 * pi

tf.random.set_seed(5)

def plot(p_gt, p_pred, rmax, results_path, timestep, quantum_numbers):
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
    fig.suptitle('{n},{l},{m} ORBITAL AT TIME t={timestep}'.format(n=n, l=l, m=m, timestep=timestep))
    plt.savefig(results_path + '{n}{l}{m}_time{timestep}.png'.format(n=n, l=l, m=m, timestep=timestep))

def solution_prior_210(args, wavefunction):
    n = 2
    Z = 1
    E_n = - (electron_mass* Z**2 * e**4) / (8 * epsilon_0**2 * h**2 * n**2)

    r, theta, phi, t = args[:,0:1], args[:,1:2], args[:,2:3], args[:,3:4]
    u, v = wavefunction[:,0:1], wavefunction[:,1:2]
    alpha = 1 / (4*np.sqrt(2*pi))
    beta = alpha*r*tf.math.exp(-r/(2*a0)) / (a0**(5/2))
    sol = beta * tf.cos(theta)

    time_factor_u = tf.cos(E_n * t / hbar)
    time_factor_v = -tf.sin(E_n * t / hbar)

    sol_u = sol * time_factor_u
    sol_v = sol * time_factor_v

    return [sol_u - u, sol_v - v]

def initial_prior_210(args, wavefunction):
    n = 2
    Z = 1
    E_n = - (electron_mass* Z**2 * e**4) / (8 * epsilon_0**2 * h**2 * n**2)

    r, theta, phi, t = args[:,0:1], args[:,1:2], args[:,2:3], args[:,3:4]
    u, v = wavefunction[:,0:1], wavefunction[:,1:2]
    alpha = 1 / (4*np.sqrt(2*pi))
    beta = alpha*r*tf.math.exp(-r/(2*a0)) / (a0**(5/2))
    sol = beta * tf.cos(theta)

    sol_u = sol
    sol_v = 0

    return [sol_u - u, sol_v - v]

def pde_polar(args, wavefunction, **pde_extra_arguments):
    '''
    args = (r, theta, phi, t)
    wavefunction = (u, v), real and imaginary parts!
    '''
    quantum_numbers = pde_extra_arguments["quantum_numbers"]
    n,l,m = quantum_numbers.values()
    
    # constants
    Z = 1
    A = l * (l+1)
    E_n = - (electron_mass* Z**2 * e**4) / (8 * epsilon_0**2 * h**2 * n**2)

    r, theta, phi, t = args[:,0:1], args[:,1:2], args[:,2:3], args[:,3:4]
    u, v = wavefunction[:,0:1], wavefunction[:,1:2]

    du_dt = dde.grad.jacobian(wavefunction, args, i=0, j=3)
    du_dr = dde.grad.jacobian(wavefunction, args, i=0, j=0)
    du_drr = dde.grad.hessian(wavefunction, args, component=0, i=0, j=0)
    du_dtheta = dde.grad.jacobian(wavefunction, args, i=0, j=1)
    du_dthetatheta = dde.grad.hessian(wavefunction, args, component=0, i=1, j=1)
    du_dphi = dde.grad.jacobian(wavefunction, args, i=0, j=2)
    du_dphiphi = dde.grad.hessian(wavefunction, args, component=0, i=2, j=2)

    dv_dt = dde.grad.jacobian(wavefunction, args, i=1, j=3)
    dv_dr = dde.grad.jacobian(wavefunction, args, i=1, j=0)
    dv_drr = dde.grad.hessian(wavefunction, args, component=1, i=0, j=0)
    dv_dtheta = dde.grad.jacobian(wavefunction, args, i=1, j=1)
    dv_dthetatheta = dde.grad.hessian(wavefunction, args, component=1, i=1, j=1)
    dv_dphi = dde.grad.jacobian(wavefunction, args, i=1, j=2)
    dv_dphiphi = dde.grad.hessian(wavefunction, args, component=1, i=2, j=2)

    ## TISE ##
    c1 = - (hbar**2 * r**2 / (2*electron_mass)) * du_drr
    c2 = - (hbar**2 * r / (electron_mass)) * du_dr
    c3 = (hbar**2 *l*(l+1) / (2*electron_mass)) * u
    c4 = (- Z *e**2 * r / (4*pi*epsilon_0)) * u
    c_time = hbar * dv_dt * r**2
    ex1 = c1+c2+c3+c4+c_time

    c1v = - (hbar**2 * r**2 / (2*electron_mass)) * dv_drr
    c2v = - (hbar**2 * r / (electron_mass)) * dv_dr
    c3v = (hbar**2 *l*(l+1) / (2*electron_mass)) * v
    c4v = (- Z *e**2 * r / (4*pi*epsilon_0)) * v
    c_timev = - hbar * du_dt * r**2
    ex2 = c1v+c2v+c3v+c4v+c_timev

    ## L2 = hbar^2 l (l+1)
    L2_u = - tf.sin(theta) * tf.cos(theta) * du_dtheta - tf.sin(theta)**2 * du_dthetatheta - du_dphiphi # we omitted hbar
    L2_v = - tf.sin(theta) * tf.cos(theta) * dv_dtheta - tf.sin(theta)**2 * dv_dthetatheta - dv_dphiphi
    ex3 = L2_u - l*(l+1)*u *tf.sin(theta)**2 # tf.sin(theta)**2 factor due to definition of the L2
    ex4 = L2_v - l*(l+1)*v *tf.sin(theta)**2

    ## Lz = hbar m
    ex5 = du_dphi + m*v
    ex6 = dv_dphi - m*u

    return [ex1 , ex2, ex3, ex4, ex5, ex6]

def create_model(experiment_params=None, network_params=None):
    assert experiment_params is not None and network_params is not None, "No parameters given!"

    quantum_numbers = experiment_params["quantum_numbers"]
    radial_extent = experiment_params["radial_extent"]
    temporal_extent = experiment_params["temporal_extent"]

    n,l,m = quantum_numbers.values()
    E_n = - (electron_mass * e**4) / (8 * epsilon_0**2 * h**2 * n**2)
    period = np.abs(2*pi*hbar / E_n)

    pde_extra_arguments = {"quantum_numbers": quantum_numbers}
    # ---------------------------------
    # geom = dde.geometry.Cuboid(xmin=[0,0,0], xmax=[30*a0, np.pi, 2*np.pi])
    # def boundary_right(x, on_boundary):
    #     return on_boundary and np.isclose(x[0], 30*a0, atol=a0*1e-08)
    geom = dde.geometry.Cuboid(xmin=[0,0,0], xmax=[radial_extent, np.pi, 2*np.pi])
    timedomain = dde.geometry.TimeDomain(0,period * temporal_extent)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    def boundary_right(args, on_boundary):
        return on_boundary and np.isclose(args[0], radial_extent)
    def g_boundary_left(args, on_boundary):
        return on_boundary and np.isclose(args[2], 0)
    def ic_func_u(args):
        r, theta, phi = args[:,0:1], args[:,1:2], args[:,2:3]
        n, l, m = quantum_numbers.values()
        R, f, g = TISE_hydrogen_exact(r, theta, phi, n,l,m)
        return R*f*g
    def ic_func_v(args):
        r, theta, phi = args[:,0:1], args[:,1:2], args[:,2:3]
        n, l, m = quantum_numbers.values()
        R, f, g = TISE_hydrogen_exact(r, theta, phi, n,l,m)
        g = np.sin(m*phi) / np.sqrt(2*pi)
        return R*f*g
    def u_func(x):
        r, theta, phi, t = x[:,0:1], x[:,1:2], x[:,2:3], x[:,3:4]
        n, l, m = quantum_numbers.values()
        R, f, g = TISE_hydrogen_exact(r, theta, phi, n,l,m)
        return R*f*g*np.cos(E_n*t/hbar)
    def v_func(x):
        r, theta, phi, t = x[:,0:1], x[:,1:2], x[:,2:3], x[:,3:4]
        n, l, m = quantum_numbers.values()
        R, f, g = TISE_hydrogen_exact(r, theta, phi, n,l,m)
        return -R*f*g*np.sin(E_n*t/hbar)
    bc_strict_u = dde.DirichletBC(geomtime, u_func, lambda _, on_boundary: on_boundary, component=0)
    bc_strict_v = dde.DirichletBC(geomtime, v_func, lambda _, on_boundary: on_boundary, component=1)
    bc_u = dde.DirichletBC(geomtime, lambda x:0, boundary_right, component=0)
    bc_v = dde.DirichletBC(geomtime, lambda x:0, boundary_right, component=1)
    bc_g_u = PeriodicBC(geomtime, 2, g_boundary_left, periodicity="symmetric", component=0)
    bc_g_v = PeriodicBC(geomtime, 2, g_boundary_left, periodicity="symmetric", component=1)
    ic_u = dde.IC(geomtime, ic_func_u, lambda _, on_initial: on_initial)
    ic_v = dde.IC(geomtime, ic_func_v, lambda _, on_initial: on_initial, component=1)
    #data = CustomTimePDE(geomtime, pde_polar, ic_bcs=[bc_u, bc_v, bc_g_u, bc_g_v, ic_u, ic_v], \
    #    num_domain=15000, num_boundary=6000, num_initial=6000, pde_extra_arguments=pde_extra_arguments)
    # TODO: uncomment the following line for strict boundary conditions
    data = CustomTimePDE(geomtime, pde_polar, ic_bcs=[bc_u, bc_v, bc_g_u, bc_g_v, ic_u, ic_v, bc_strict_u, bc_strict_v], \
        num_domain=network_params["num_domain"], num_boundary=network_params["num_domain"], \
             num_initial=network_params["num_initial"], pde_extra_arguments=pde_extra_arguments)

    if network_params["backbone"] == "FNN":
        #net = dde.maps.FNN([3] + [50] * 4 + [2], "tanh", "Glorot normal")
        net = dde.maps.FNN(network_params["layers"], "tanh", "Glorot normal")
    elif network_params["backbone"] == "ResNet":
        input_size = network_params["input_size"]
        output_size = network_params["output_size"]
        num_neurons = network_params["num_neurons"]
        num_blocks = network_params["num_blocks"]
        net = dde.maps.ResNet(input_size, output_size, num_neurons, num_blocks, "tanh", "Glorot normal")
    else:
        raise NotImplementedError("Only FNN backbone is experiment ready!")

    model = dde.Model(data, net)
    return model, geomtime, data, net

def normalize_output(model, experiment_params):
    radial_extent = experiment_params["radial_extent"]

    from scipy.integrate import simps
    #rmax = 20
    rmax = np.sqrt((radial_extent**2) / 3)
    n_points = 100 # must be an even number
    x = np.linspace(-rmax,rmax,n_points)
    y = np.linspace(-rmax,rmax,n_points)
    z = np.linspace(-rmax,rmax,n_points)
    X, Y, Z = np.meshgrid(x, y, z)
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arctan(np.sqrt(X**2 + Y**2) / Z)
    theta = np.where(theta<0,np.pi+theta,theta)
    phi = np.where(X<0, np.arctan(Y/X), np.arctan(Y/X)+pi)

    r = r.reshape((n_points**3, 1))
    theta = theta.reshape((n_points**3, 1))
    phi = phi.reshape((n_points**3, 1))
    time = np.zeros_like(r) # normalize at initial point
    input = np.hstack((r, theta, phi, time))
    predictions_u = model.predict(input)[:,0]
    predictions_v = model.predict(input)[:,1]
    predictions_u = predictions_u.reshape((n_points, n_points, n_points))
    predictions_v = predictions_v.reshape((n_points, n_points, n_points))

    #integrand = predictions_u**2 # if we assume Im(wavefunction)=0
    integrand = predictions_u**2 + predictions_v**2
    integral = simps(integrand, z)
    integral = simps(integral, y)
    integral = simps(integral, x)
    normalization_costant = 1. / np.sqrt(integral)
    return normalization_costant

def main(config_path=None):

    if config_path is None:
        root = "/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/"
        config_path = root + "experiment_configs/TDSE_unperturbed/config_210.yaml"
    
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    experiment_params = config["experiment_params"]
    network_params = config["network_params"]
    prior_params = config["prior_params"]
    results_path = config["results_path"]
    #results_path = '/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/results_TDSE_unperturbed/' # path in local
    
    quantum_numbers = experiment_params["quantum_numbers"] # quantum numbers of the initial state
    
    folder = str(quantum_numbers["n"]) + str(quantum_numbers["l"]) + str(quantum_numbers["m"]) + '/'
    results_path = results_path + folder
    import os
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    pde_extra_arguments = {"quantum_numbers":quantum_numbers}
    model, geomtime, data, net = create_model(experiment_params, network_params)

    # # PRIOR LEARING OF INITIAL FUNCTION # TODO: add prior params
    # prior_data = CustomTimePDE(geomtime, initial_prior_210, ic_bcs=[], \
    #     num_domain=15000, num_boundary=6000, num_initial=6000)
    # prior_save_path = results_path + "210_prior/model"
    # compile_train_args = {'optimizer':'adam', 'lr':1e-3, 'epochs':20000}
    # model = CustomLossModel(data, net)
    # model.learn_prior(prior_data, prior_save_path, **compile_train_args)

    # MAIN EXPERIMENTS: decreasing learning rate
    # TODO: comment out model restoring when running on cluster
    assert len(network_params["lrs"]) == len(network_params["optimizers"]) and \
        len(network_params["lrs"]) == len(network_params["epochs"]), "Incompatible network parameter lengths!"
    for i in range(len(network_params["lrs"])):
        lr = float(network_params["lrs"][i])
        optimizer = network_params["optimizers"][i]
        epoch = network_params["epochs"][i]
        model.compile(optimizer=optimizer, lr=lr, loss="MSE", loss_weights=network_params["loss_weights"])
        # with open(results_path+"model/best_step.txt") as f: # TODO: add this after first loop
        #     best = f.readlines()[0]
        # model.restore(results_path + "model/model.ckpt-" + best, verbose=1)
        checker = dde.callbacks.ModelCheckpoint(
            results_path+"model/model.ckpt", save_better_only=True, period=1000
        )
        losshistory, train_state = model.train(epochs=epoch, callbacks=[checker])
        with open(results_path+"model/best_step.txt", "w") as text_file:
            text_file.write(str(train_state.best_step))

    # ---------------------------------
    # # PRIOR CONVERGENCE TEST
    # prior_data = CustomTimePDE(geomtime, solution_prior_210, ic_bcs=[], \
    #     num_domain=1500, num_boundary=600, num_initial=600)
    # prior_save_path = results_path + "210_prior/model"
    # compile_train_args = {'optimizer':'adam', 'lr':1e-4, 'epochs':30000}
    # model = CustomLossModel(data, net)
    # model.learn_prior(prior_data, prior_save_path, **compile_train_args)
    # model.compile("adam", lr=1.0e-5, loss='MSE')
    # ---------------------------------
    #model.train(epochs=10000) # TODO: uncomment when NOT running MAIN EXPERIMENTS

    # X = geom.uniform_points(1000000)
    # wavefunc_pred = model.predict(X)


    # ## RESTORE BEST STEP ## TODO: uncomment when running MAIN EXPERIMENTS
    # tf.compat.v1.reset_default_graph()
    # model = create_model(quantum_numbers)
    # model.compile("adam", lr=1.0e-3)
    # with open(results_path+"model/best_step.txt", "r") as text_file:
    #     best = text_file.readlines()[0]
    # model.restore(results_path+"model/model.ckpt-" + best, verbose=1)

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

    r = r.reshape((n_points*n_points, 1))
    theta = theta.reshape((n_points*n_points, 1))
    phi = phi.reshape((n_points*n_points, 1))

    E_n = - (electron_mass * e**4) / (8 * epsilon_0**2 * h**2 * n**2)
    period = np.abs(2*pi*hbar / E_n)
    time_points = [x*period/4 for x in range(4)]

    for time_point in time_points:
        time = time_point * np.ones_like(r)
        input_format = np.hstack((r, theta, phi, time))
        wavefunction_pred = model.predict(input_format)
        wavefunction_pred_real = wavefunction_pred[:,0:1] # real part
        wavefunction_pred_imag = wavefunction_pred[:,1:2] # imaginary part
        # normalization
        normalization_constant = normalize_output(model=model, experiment_params=experiment_params)
        wavefunction_pred_real = normalization_constant * wavefunction_pred_real
        wavefunction_pred_imag = normalization_constant * wavefunction_pred_imag
        ###############
        wavefunction_pred_real = wavefunction_pred_real.reshape((n_points, n_points))
        wavefunction_pred_imag = wavefunction_pred_imag.reshape((n_points, n_points))
        p_pred = wavefunction_pred_real**2 + wavefunction_pred_imag**2

        time = time.reshape((n_points, n_points))
        wavefunction_gt_real = wavefunction_gt * np.cos(E_n * time / hbar)
        wavefunction_gt_imag = - wavefunction_gt * np.sin(E_n * time / hbar)
        p_gt = wavefunction_gt_real**2 + wavefunction_gt_imag**2

        plot(p_gt, p_pred, rmax, results_path, time_point, quantum_numbers)


if __name__ == "__main__":

    #root = "/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/"
    root = "./"
    default_config_path = root + "experiment_configs/TDSE_unperturbed/config_210.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, \
        default=default_config_path ,help='Path to the experiment config file.')
    args = parser.parse_args()
    config_path = args.config_path

    main(config_path)