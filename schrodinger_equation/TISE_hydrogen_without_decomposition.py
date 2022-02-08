import yaml
import argparse

from deepxde import backend
import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import matplotlib.pyplot as plt
import deepxde as dde
from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass, physical_constants
a0 = physical_constants["Bohr radius"][0]
from pinnse.schrodinger_eq_exact_values import TISE_hydrogen_exact

from pinnse.custom_model import CustomLossModel
from pinnse.custom_boundary_conditions import PeriodicBC
from pinnse.custom_pde import CustomPDE
from pinnse.basis_functions import FourierBasis
from matplotlib import cm

# ATOMIC UNITS
hbar = 1
a0 = 1
e = 1
electron_mass = 1
epsilon_0 = 1 / (4*pi)
h = hbar * 2 * pi

tf.random.set_seed(5)

def solution_prior_321(args, wavefunction):
    r, theta, phi = args[:,0:1], args[:,1:2], args[:,2:3]
    u, v = wavefunction[:,0:1], wavefunction[:,1:2]
    r = r/a0
    sol_u = tf.cos(phi) * tf.cos(theta) * tf.sin(theta) * (r**2) * tf.math.exp(-r/3)
    sol_v = tf.sin(phi) * tf.cos(theta) * tf.sin(theta) * (r**2) * tf.math.exp(-r/3)

    du_dr = dde.grad.jacobian(wavefunction, args, i=0, j=0)
    du_drr = dde.grad.hessian(wavefunction, args, component=0, i=0, j=0)
    du_dtheta = dde.grad.jacobian(wavefunction, args, i=0, j=1)
    du_dthetatheta = dde.grad.hessian(wavefunction, args, component=0, i=1, j=1)
    du_dphi = dde.grad.jacobian(wavefunction, args, i=0, j=2)
    du_dphiphi = dde.grad.hessian(wavefunction, args, component=0, i=2, j=2)

    dv_dr = dde.grad.jacobian(wavefunction, args, i=1, j=0)
    dv_drr = dde.grad.hessian(wavefunction, args, component=1, i=0, j=0)
    dv_dtheta = dde.grad.jacobian(wavefunction, args, i=1, j=1)
    dv_dthetatheta = dde.grad.hessian(wavefunction, args, component=1, i=1, j=1)
    dv_dphi = dde.grad.jacobian(wavefunction, args, i=1, j=2)
    dv_dphiphi = dde.grad.hessian(wavefunction, args, component=1, i=2, j=2)

    a = tf.cos(phi) * tf.cos(theta) * tf.sin(theta)
    b = tf.sin(phi) * tf.cos(theta) * tf.sin(theta)
    c = (r**2) * tf.math.exp(-r/3)
    sol_u_dr = a * (2*r*tf.math.exp(-r/3) - r**2 * tf.math.exp(-r/3) / 3)
    sol_u_drr = a * (2*tf.math.exp(-r/3) - 4*r*tf.math.exp(-r/3)/3 + r**2 * tf.math.exp(-r/3)/9)
    sol_u_dtheta = tf.cos(phi) * (tf.cos(theta)**2 - tf.sin(theta)**2) * c
    sol_u_dthetatheta = tf.cos(phi) * (-4*tf.cos(theta)*tf.sin(theta)) * c
    sol_u_dphi = -tf.sin(phi)*tf.cos(theta)*tf.sin(theta)*c
    sol_u_dphiphi = -tf.cos(phi)*tf.cos(theta)*tf.sin(theta)*c

    sol_v_dr = b * (2*r*tf.math.exp(-r/3) - r**2 * tf.math.exp(-r/3) / 3)
    sol_v_drr = b * (2*tf.math.exp(-r/3) - 4*r*tf.math.exp(-r/3)/3 + r**2 * tf.math.exp(-r/3)/9)
    sol_v_dtheta = tf.sin(phi) * (tf.cos(theta)**2 - tf.sin(theta)**2) * c
    sol_v_dthetatheta = tf.sin(phi) * (-4*tf.cos(theta)*tf.sin(theta)) * c
    sol_v_dphi = tf.cos(phi)*tf.cos(theta)*tf.sin(theta)*c
    sol_v_dphiphi = -tf.sin(phi)*tf.cos(theta)*tf.sin(theta)*c

    ex1,ex2,ex3,ex4,ex5,ex6 = sol_u_dr-du_dr, sol_u_drr-du_drr, sol_u_dtheta-du_dtheta, sol_u_dthetatheta-du_dthetatheta, sol_u_dphi-du_dphi, sol_u_dphiphi-du_dphiphi
    ex7,ex8,ex9,ex10,ex11,ex12 = sol_v_dr-dv_dr, sol_v_drr-dv_drr, sol_v_dtheta-dv_dtheta, sol_v_dthetatheta-dv_dthetatheta, sol_v_dphi-dv_dphi, sol_v_dphiphi-dv_dphiphi

    return [sol_u - u, sol_v - v, ex1,ex2,ex3,ex4,ex5,ex6, ex7,ex8,ex9,ex10,ex11,ex12]
    #return [sol_u - u, sol_v - v]


def solution_prior_210(args, wavefunction):
    r, theta, phi = args[:,0:1], args[:,1:2], args[:,2:3]
    u, v = wavefunction[:,0:1], wavefunction[:,1:2]
    alpha = 1 / (4*np.sqrt(2*pi))
    beta = alpha*r*tf.math.exp(-r/(2*a0)) / (a0**(5/2))
    sol_u = beta * tf.cos(theta)
    sol_v = 0

    du_dr = dde.grad.jacobian(wavefunction, args, i=0, j=0)
    du_drr = dde.grad.hessian(wavefunction, args, component=0, i=0, j=0)
    du_dtheta = dde.grad.jacobian(wavefunction, args, i=0, j=1)
    du_dthetatheta = dde.grad.hessian(wavefunction, args, component=0, i=1, j=1)
    du_dphi = dde.grad.jacobian(wavefunction, args, i=0, j=2)
    du_dphiphi = dde.grad.hessian(wavefunction, args, component=0, i=2, j=2)

    dv_dr = dde.grad.jacobian(wavefunction, args, i=1, j=0)
    dv_drr = dde.grad.hessian(wavefunction, args, component=1, i=0, j=0)
    dv_dtheta = dde.grad.jacobian(wavefunction, args, i=1, j=1)
    dv_dthetatheta = dde.grad.hessian(wavefunction, args, component=1, i=1, j=1)
    dv_dphi = dde.grad.jacobian(wavefunction, args, i=1, j=2)
    dv_dphiphi = dde.grad.hessian(wavefunction, args, component=1, i=2, j=2)

    sol_u_dr = (alpha * tf.cos(theta) * tf.math.exp(-r/(2*a0)) / (a0**(5/2))) - (alpha *tf.cos(theta)*r*tf.math.exp(-r/(2*a0)) / (2*(a0**(7/2))))
    sol_u_drr = (-alpha * tf.cos(theta) * tf.math.exp(-r/(2*a0)) / (a0**(7/2))) + (alpha*tf.cos(theta)*r*tf.math.exp(-r/(2*a0)) / (4*(a0**(9/2))))
    sol_u_dtheta = beta * (-tf.sin(theta))
    sol_u_dthetatheta = beta * (-tf.cos(theta))
    sol_u_dphi = 0
    sol_u_dphiphi = 0

    sol_v_dr = 0
    sol_v_drr = 0
    sol_v_dtheta = 0
    sol_v_dthetatheta = 0
    sol_v_dphi = 0
    sol_v_dphiphi = 0

    ex1,ex2,ex3,ex4,ex5,ex6 = sol_u_dr-du_dr, sol_u_drr-du_drr, sol_u_dtheta-du_dtheta, sol_u_dthetatheta-du_dthetatheta, sol_u_dphi-du_dphi, sol_u_dphiphi-du_dphiphi
    ex7,ex8,ex9,ex10,ex11,ex12 = sol_v_dr-dv_dr, sol_v_drr-dv_drr, sol_v_dtheta-dv_dtheta, sol_v_dthetatheta-dv_dthetatheta, sol_v_dphi-dv_dphi, sol_v_dphiphi-dv_dphiphi

    return [sol_u - u, sol_v - v, ex1,ex2,ex3,ex4,ex5,ex6, ex7,ex8,ex9,ex10,ex11,ex12]


def fourier_prior(args, f):
    basis = FourierBasis(max_k=10, dimension=3, axis=1) # axis=1 corresponds to the polar variable theta
    representation = basis.compute(args)
    return representation - f


def pde(args, wavefunction, **quantum_numbers): # TODO: have not been tested yet, implementation in cartesian coordinates
    n,l,m = quantum_numbers.values()
    
    # constants
    Z = 1
    A = l * (l+1)
    E_n = - (electron_mass* Z**2 * e**4) / (8 * epsilon_0**2 * h**2 * n**2)

    x,y,z = args[:,0:1], args[:,1:2], args[:,2:3]
    u, v = wavefunction[:,0:1], wavefunction[:,1:2]
    r = tf.sqrt(x**2 + y**2 + z**2)

    du_dxx = dde.grad.hessian(wavefunction, args, component=0, i=0 ,j=0)
    du_dyy = dde.grad.hessian(wavefunction, args, component=0, i=1 ,j=1)
    du_dzz = dde.grad.hessian(wavefunction, args, component=0, i=2 ,j=2)

    dv_dxx = dde.grad.hessian(wavefunction, args, component=1, i=0 ,j=0)
    dv_dyy = dde.grad.hessian(wavefunction, args, component=1, i=1 ,j=1)
    dv_dzz = dde.grad.hessian(wavefunction, args, component=1, i=2 ,j=2)

    laplace_u = du_dxx + du_dyy + du_dzz
    laplace_v = dv_dxx + dv_dyy + dv_dzz

    ex1 = - (hbar**2 / (2*electron_mass)) * laplace_u *r - (Z*e**2 / (4*pi*epsilon_0)) * u - E_n * u *r # we multiplied by r (to get rid of singularities)
    ex2 = - (hbar**2 / (2*electron_mass)) * laplace_v *r - (Z*e**2 / (4*pi*epsilon_0)) * v - E_n * v *r

    return [ex1, ex2]


def pde_polar(args, wavefunction, **quantum_numbers):
    '''
    args = (r, theta, phi)
    wavefunction = (u, v), real and imaginary parts!
    '''
    n,l,m = quantum_numbers.values()
    
    # constants
    Z = 1
    A = l * (l+1)
    E_n = - (electron_mass* Z**2 * e**4) / (8 * epsilon_0**2 * h**2 * n**2)

    r, theta, phi = args[:,0:1], args[:,1:2], args[:,2:3]
    u, v = wavefunction[:,0:1], wavefunction[:,1:2]

    du_dr = dde.grad.jacobian(wavefunction, args, i=0, j=0)
    du_drr = dde.grad.hessian(wavefunction, args, component=0, i=0, j=0)
    du_dtheta = dde.grad.jacobian(wavefunction, args, i=0, j=1)
    du_dthetatheta = dde.grad.hessian(wavefunction, args, component=0, i=1, j=1)
    du_dphi = dde.grad.jacobian(wavefunction, args, i=0, j=2)
    du_dphiphi = dde.grad.hessian(wavefunction, args, component=0, i=2, j=2)

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
    c5 = (electron_mass * Z**2 * e**4 * r**2 / (8*epsilon_0**2 *h**2 * n**2)) * u
    ex1 = c1+c2+c3+c4+c5
    c1v = - (hbar**2 * r**2 / (2*electron_mass)) * dv_drr
    c2v = - (hbar**2 * r / (electron_mass)) * dv_dr
    c3v = (hbar**2 *l*(l+1) / (2*electron_mass)) * v
    c4v = (- Z *e**2 * r / (4*pi*epsilon_0)) * v
    c5v = (electron_mass * Z**2 * e**4 * r**2 / (8*epsilon_0**2 *h**2 * n**2)) * v
    ex2 = c1v+c2v+c3v+c4v+c5v

    ## L2 = hbar^2 l (l+1)
    L2_u = - tf.sin(theta) * tf.cos(theta) * du_dtheta - tf.sin(theta)**2 * du_dthetatheta - du_dphiphi # we omitted hbar
    L2_v = - tf.sin(theta) * tf.cos(theta) * dv_dtheta - tf.sin(theta)**2 * dv_dthetatheta - dv_dphiphi
    ex3 = L2_u - l*(l+1)*u *tf.sin(theta)**2 # tf.sin(theta)**2 factor due to definition of the L2
    ex4 = L2_v - l*(l+1)*v *tf.sin(theta)**2

    ## Lz = hbar m
    ex5 = - du_dphi - m*v
    ex6 = dv_dphi - m*u

    return [ex1 , ex2, ex3, ex4, ex5, ex6]


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
    input = np.hstack((r, theta, phi))
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


def create_model(experiment_params=None, network_params=None):
    assert experiment_params is not None and network_params is not None, "No parameters given!"

    quantum_numbers = experiment_params["quantum_numbers"]
    radial_extent = experiment_params["radial_extent"]

    # ---------------------------------
    # geom = dde.geometry.Cuboid(xmin=[0,0,0], xmax=[30*a0, np.pi, 2*np.pi])
    # def boundary_right(x, on_boundary):
    #     return on_boundary and np.isclose(x[0], 30*a0, atol=a0*1e-08)
    geom = dde.geometry.Cuboid(xmin=[0,0,0], xmax=[radial_extent, np.pi, 2*np.pi])
    def boundary_right(x, on_boundary):
        return on_boundary and np.isclose(x[0], radial_extent)
    def g_boundary_left(x, on_boundary):
        return on_boundary and np.isclose(x[2], 0)
    def u_func(x):
        r, theta, phi = x[:,0:1], x[:,1:2], x[:,2:3]
        n, l, m = quantum_numbers.values()
        R, f, g = TISE_hydrogen_exact(r, theta, phi, n,l,m)
        return R*f*g
    def v_func(x):
        r, theta, phi = x[:,0:1], x[:,1:2], x[:,2:3]
        n, l, m = quantum_numbers.values()
        R, f, g = TISE_hydrogen_exact(r, theta, phi, n,l,m)
        g = np.sin(m*phi) / np.sqrt(2*pi)
        return R*f*g
    bc_strict_u = dde.DirichletBC(geom, u_func, lambda _, on_boundary: on_boundary, component=0)
    bc_strict_v = dde.DirichletBC(geom, v_func, lambda _, on_boundary: on_boundary, component=1)
    bc_u = dde.DirichletBC(geom, lambda x:0, boundary_right, component=0)
    bc_v = dde.DirichletBC(geom, lambda x:0, boundary_right, component=1)
    bc_g_u = PeriodicBC(geom, 2, g_boundary_left, periodicity="symmetric", component=0)
    bc_g_v = PeriodicBC(geom, 2, g_boundary_left, periodicity="symmetric", component=1)
    #data = CustomPDE(geom, pde_polar, bcs=[bc_u, bc_v, bc_g_u, bc_g_v], num_domain=1500, num_boundary=600, pde_extra_arguments=quantum_numbers)
    # TODO: uncomment the following line for strict boundary conditions
    data = CustomPDE(geom, pde_polar, bcs=[bc_strict_u, bc_strict_v, bc_u, bc_v, bc_g_u, bc_g_v],\
         num_domain=network_params["num_domain"], num_boundary=network_params["num_boundary"], pde_extra_arguments=quantum_numbers)
    # ---------------------------------
    # geom = dde.geometry.Cuboid(xmin=[-20,-20,-20], xmax=[20, 20, 20]) # TODO: setup for cartesian coordinates
    # def boundary_right_x(x, on_boundary):
    #     return on_boundary and np.isclose(x[0], 20)
    # def boundary_right_y(x, on_boundary):
    #     return on_boundary and np.isclose(x[1], 20)
    # def boundary_right_z(x, on_boundary):
    #     return on_boundary and np.isclose(x[2], 20)
    # bc1 = dde.DirichletBC(geom, lambda x:0, boundary_right_x)
    # bc2 = dde.DirichletBC(geom, lambda x:0, boundary_right_y)
    # bc3 = dde.DirichletBC(geom, lambda x:0, boundary_right_z)
    # data = CustomPDE(geom, pde, bcs=[bc1, bc2, bc3], num_domain=1500, num_boundary=600, pde_extra_arguments=quantum_numbers)
    # ---------------------------------

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
    return model, geom, data, net


def main(config_path=None):
    
    if config_path is None:
        root = "/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/"
        config_path = root + "experiment_configs/TISE_hydrogen_without_decomposition/config_210.yaml"
    
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    experiment_params = config["experiment_params"]
    network_params = config["network_params"]
    prior_params = config["prior_params"]
    results_path = config["results_path"]
    #results_path = '/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/results_TISE_hydrogen_without_decomposition/' # path in local

    quantum_numbers = experiment_params["quantum_numbers"]

    folder = str(quantum_numbers["n"]) + str(quantum_numbers["l"]) + str(quantum_numbers["m"]) + '/'
    results_path = results_path + folder
    import os
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    model, geom, data, net = create_model(experiment_params, network_params)

    # MAIN EXPERIMENTS: decreasing learning rate
    # TODO: comment out model restoring when running on cluster
    assert len(network_params["lrs"]) == len(network_params["optimizers"]) and \
        len(network_params["lrs"]) == len(network_params["epochs"]), "Incompatible network parameter lengths!"
    for i in range(len(network_params["lrs"])):
        lr = float(network_params["lrs"][i])
        optimizer = network_params["optimizers"][i]
        epoch = network_params["epochs"][i]
        model.compile(optimizer=optimizer, lr=lr, loss_weights=network_params["loss_weights"])
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
    # model = CustomLossModel(data, net)
    # model.compile("adam", lr=1.0e-3, loss="NormalizationLoss")
    # ---------------------------------
    # # PRIOR CONVERGENCE TEST
    # prior_data = dde.data.PDE(geom, solution_prior_210, bcs=[], num_domain=1500, num_boundary=600)
    # prior_save_path = results_path + "210_prior/model"
    # compile_train_args = {'optimizer':'adam', 'lr':1e-3, 'epochs':10000}
    # model = CustomLossModel(data, net)
    # model.learn_prior(prior_data, prior_save_path, **compile_train_args)
    # model.compile("adam", lr=1.0e-5, loss='MSE')
    # #model.compile("adam", lr=1.0e-5, loss='NormalizationLoss')
    # ---------------------------------
    # # FOURIER PRIOR ON THETA: TODO poor results
    # prior_data = dde.data.PDE(geom, fourier_prior, bcs=[], num_domain=1500, num_boundary=600)
    # prior_save_path = results_path + "fourier_prior/model"
    # compile_train_args = {'optimizer':'adam', 'lr':1e-3, 'epochs':10000}
    # model = CustomLossModel(data, net)
    # model.learn_prior(prior_data, prior_save_path, **compile_train_args)
    # model.compile("adam", lr=1.0e-5, loss='MSE')
    # #model.compile("adam", lr=1.0e-5, loss='NormalizationLoss')
    # ---------------------------------
    #model.train(epochs=20000) # TODO: uncomment when not running MAIN EXPERIMENTS

    # X = geom.uniform_points(1000000)
    # wavefunc_pred = model.predict(X)


    # ## RESTORE BEST STEP ##
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
    p_gt = wavefunction_gt**2 # Probability distribution

    r = r.reshape((n_points*n_points, 1))
    theta = theta.reshape((n_points*n_points, 1))
    phi = phi.reshape((n_points*n_points, 1))
    input_format = np.hstack((r, theta, phi))
    wavefunction_pred = model.predict(input_format)
    wavefunction_pred = wavefunction_pred[:,0:1] # real part
    # normalization
    normalization_constant = normalize_output(model=model, experiment_params=experiment_params)
    wavefunction_pred = normalization_constant * wavefunction_pred
    ###############
    wavefunction_pred = wavefunction_pred.reshape((n_points, n_points))
    p_pred = wavefunction_pred**2

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


if __name__ == "__main__":

    #root = "/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/"
    root = "./"
    default_config_path = root + "experiment_configs/TISE_hydrogen_without_decomposition/config_210.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, \
        default=default_config_path ,help='Path to the experiment config file.')
    args = parser.parse_args()
    config_path = args.config_path

    main(config_path)