import yaml
import argparse

from deepxde import backend
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepxde as dde
from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass, physical_constants
a0 = physical_constants["Bohr radius"][0]
from scipy.integrate import simps
from pinnse.schrodinger_eq_exact_values import TISE_hydrogen_exact, TISE_stark_effect_exact

from pinnse.custom_model import CustomLossModel
from pinnse.custom_boundary_conditions import PeriodicBC
from pinnse.custom_pde import CustomPDE
from pinnse.basis_functions import FourierBasis

# ATOMIC UNITS
hbar = 1
a0 = 1
e = 1
electron_mass = 1
epsilon_0 = 1 / (4*pi)
h = hbar * 2 * pi

tf.random.set_seed(5)

def fourier_prior(args, f):
    x = args[:,0:1]
    y = f[:,0:1]
    basis = FourierBasis(max_k=10)
    representation = basis.compute(x)
    return representation - y


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

    #return [sol_u - u, sol_v - v, ex1,ex2,ex3,ex4,ex5,ex6, ex7,ex8,ex9,ex10,ex11,ex12]
    return [sol_u - u, sol_v - v]


def solution_prior_20index0(args, wavefunction):
    r, theta, phi = args[:,0:1], args[:,1:2], args[:,2:3]
    u, v = wavefunction[:,0:1], wavefunction[:,1:2]
    alpha = 1 / (4*np.sqrt(2*pi))
    beta = alpha*r*tf.math.exp(-r/(2*a0)) / (a0**(5/2))
    sol_w = beta * tf.cos(theta) # 210
    sol_z = alpha * (1/(a0**(3/2))) * (2 - (r/a0)) * tf.math.exp(-r/(2*a0)) # 200
    sol_u = (1/np.sqrt(2)) * (sol_w + sol_z)
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

    sol_w_dr = (alpha * tf.cos(theta) * tf.math.exp(-r/(2*a0)) / (a0**(5/2))) - (alpha *tf.cos(theta)*r*tf.math.exp(-r/(2*a0)) / (2*(a0**(7/2))))
    sol_w_drr = (-alpha * tf.cos(theta) * tf.math.exp(-r/(2*a0)) / (a0**(7/2))) + (alpha*tf.cos(theta)*r*tf.math.exp(-r/(2*a0)) / (4*(a0**(9/2))))
    sol_w_dtheta = beta * (-tf.sin(theta))
    sol_w_dthetatheta = beta * (-tf.cos(theta))
    sol_w_dphi = 0
    sol_w_dphiphi = 0
    
    alpha2 = alpha * (1/(a0**(3/2)))
    c = tf.math.exp(-r/(2*a0))
    sol_z_dr = alpha2*r*c / (2*a0**2) - 2*alpha2*c/a0
    sol_z_drr = alpha2*c/(a0**2) + (alpha2/(4*a0**2)) * (2 - (r/a0))
    sol_z_dtheta = 0
    sol_z_dthetatheta = 0 
    sol_z_dphi = 0
    sol_z_dphiphi = 0

    sol_u_dr = (1/np.sqrt(2)) * (sol_w_dr + sol_z_dr)
    sol_u_drr = (1/np.sqrt(2)) * (sol_w_drr + sol_z_drr)
    sol_u_dtheta = (1/np.sqrt(2)) * (sol_w_dtheta + sol_z_dtheta)
    sol_u_dthetatheta = (1/np.sqrt(2)) * (sol_w_dthetatheta + sol_z_dthetatheta)
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

    #return [sol_u - u, sol_v - v, ex1,ex2,ex3,ex4,ex5,ex6, ex7,ex8,ex9,ex10,ex11,ex12]
    return [sol_u - u, sol_v - v]


def pde_polar(args, wavefunction, **pde_extra_arguments):
    '''
    args = (r, theta, phi)
    wavefunction = (u, v), real and imaginary parts!
    '''
    quantum_numbers = pde_extra_arguments["quantum_numbers"]
    electric_field = pde_extra_arguments["electric_field"]
    delE = pde_extra_arguments["delE"]
    #n,l,m = quantum_numbers.values()
    n,m = quantum_numbers.values() # Here we only have n,m
    
    # constants
    Z = 1
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
    c1 = - (hbar**2 * r**2 * tf.sin(theta)**2 / (2*electron_mass)) * du_drr
    c2 = - (hbar**2 * r * tf.sin(theta)**2 / (electron_mass)) * du_dr
    c3 = - (hbar**2 * tf.sin(theta)**2 / (2.*electron_mass)) * du_dthetatheta
    c4 = - (hbar**2 * tf.sin(theta) * tf.cos(theta) / (2.*electron_mass)) * du_dtheta
    c5 = - (hbar**2 / (2.*electron_mass)) * du_dphiphi
    c6 = (- Z *e**2 * r *tf.sin(theta)**2 / (4*pi*epsilon_0)) * u
    E0 = - (electron_mass * Z**2 * e**4 / (8*epsilon_0**2 *h**2 * n**2))
    E = E0 + delE
    c7 = - E * r**2 * tf.sin(theta)**2 * u
    #c7 = (electron_mass * Z**2 * e**4 * r**2 * tf.sin(theta)**2 / (8*epsilon_0**2 *h**2 * n**2)) * u
    c_electric = e * electric_field * r**3 * tf.sin(theta)**2 * tf.cos(theta) * u
    ex1 = c1+c2+c3+c4+c5+c6+c7+c_electric
    c1v = - (hbar**2 * r**2 * tf.sin(theta)**2 / (2*electron_mass)) * dv_drr
    c2v = - (hbar**2 * r * tf.sin(theta)**2 / (electron_mass)) * dv_dr
    c3v = - (hbar**2 * tf.sin(theta)**2 / (2.*electron_mass)) * dv_dthetatheta
    c4v = - (hbar**2 * tf.sin(theta) * tf.cos(theta) / (2.*electron_mass)) * dv_dtheta
    c5v = - (hbar**2 / (2.*electron_mass)) * dv_dphiphi
    c6v = (- Z *e**2 * r *tf.sin(theta)**2 / (4*pi*epsilon_0)) * v
    c7v = - E * r**2 * tf.sin(theta)**2 * v
    #c7v = (electron_mass * Z**2 * e**4 * r**2 * tf.sin(theta)**2 / (8*epsilon_0**2 *h**2 * n**2)) * v
    c_electricv = - e * electric_field * r**3 * tf.sin(theta)**2 * tf.cos(theta) * v
    ex2 = c1v+c2v+c3v+c4v+c5v+c6v+c7v+c_electricv

    ## Lz = hbar m
    ex3 = - du_dphi - m*v
    ex4 = dv_dphi - m*u

    return [ex1 , ex2, ex3, ex4]


def create_model(experiment_params=None, network_params=None):
    assert experiment_params is not None and network_params is not None, "No parameters given!"

    quantum_numbers = experiment_params["quantum_numbers"]
    electric_field = float(experiment_params["electric_field"])
    index = experiment_params["index"] # which energy to consider for given manifold
    radial_extent = experiment_params["radial_extent"]

    n, m = quantum_numbers.values()

    eigenvalues, _ = TISE_stark_effect_exact(0,0,0, electric_field, n,m)
    delE = eigenvalues[index]
    pde_extra_arguments = {"quantum_numbers":quantum_numbers, "electric_field":electric_field, "delE":delE}

    def u_func(x):
        r, theta, phi = x[:,0:1], x[:,1:2], x[:,2:3]
        _, eigenvectors = TISE_stark_effect_exact(r, theta, phi, electric_field, n,m)
        return eigenvectors[index]
    # def u_func(x): # Sanity check using TISE hydrogen solutions
    #     r, theta, phi = x[:,0:1], x[:,1:2], x[:,2:3]
    #     n, l, m = quantum_numbers.values()
    #     R, f, g = TISE_hydrogen_exact(r, theta, phi, n,l,m)
    #     return R*f*g
    def v_func(x): # TODO: generalize
        return 0
    geom = dde.geometry.Cuboid(xmin=[0,0,0], xmax=[radial_extent, np.pi, 2*np.pi])
    def boundary_right(x, on_boundary):
        return on_boundary and np.isclose(x[0], radial_extent)
    def g_boundary_left(x, on_boundary):
        return on_boundary and np.isclose(x[2], 0)
    bc_strict_u = dde.DirichletBC(geom, u_func, lambda _, on_boundary: on_boundary, component=0)
    bc_strict_v = dde.DirichletBC(geom, v_func, lambda _, on_boundary: on_boundary, component=1)
    bc_u = dde.DirichletBC(geom, lambda x:0, boundary_right, component=0)
    bc_v = dde.DirichletBC(geom, lambda x:0, boundary_right, component=1)
    bc_g_u = PeriodicBC(geom, 2, g_boundary_left, periodicity="symmetric", component=0)
    bc_g_v = PeriodicBC(geom, 2, g_boundary_left, periodicity="symmetric", component=1)
    #data = CustomPDE(geom, pde_polar, bcs=[bc_u, bc_v, bc_g_u, bc_g_v], num_domain=1500, num_boundary=600, pde_extra_arguments=quantum_numbers)
    # TODO: uncomment the following line for strict boundary conditions
    data = CustomPDE(geom, pde_polar, bcs=[bc_strict_u, bc_strict_v, bc_u, bc_v, bc_g_u, bc_g_v], num_domain=1500, num_boundary=600, pde_extra_arguments=pde_extra_arguments)
    
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

# ---------------------------------
# ---------------------------------
# ---------------------------------

# ---------------------------------
# ---------------------------------
# ---------------------------------

def main_initial():
    results_path = '/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/results_TISE_stark_effect/'
    # To be specified quantum numbers are n and m, since the manifolds are defined only by m for a given n,
    # l has then the values: m <= l <= n-1
    quantum_numbers = {'n':2, 'm':0}
    # n = 2 testing (where we have non-degenerate results): https://courses.physics.illinois.edu/phys581/sp2014/stark.pdf  

    # ----------- N=2 , M=0 -----------

    # electric_field = 10e-12 # TODO: fix this, so electric field is perturbative
    # delE_minus = - 3 * e * electric_field * a0
    # delE_plus = 3 * e * electric_field * a0
    # pde_extra_arguments = {"quantum_numbers":quantum_numbers, "electric_field":electric_field, "delE":delE_minus} # extra arguments for wavefunction_minus

    # model = create_model(pde_extra_arguments)

    # model.compile("adam", lr=1.0e-3)
    # model.train(epochs=10000)

    # ----------------------
    rmax = 30
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

    R_nl_210, f_lm_210, g_m_210 = TISE_hydrogen_exact(r, theta, phi,
         quantum_numbers['n'] , 1, quantum_numbers['m'])
    R_nl_200, f_lm_200, g_m_200 = TISE_hydrogen_exact(r, theta, phi,
         quantum_numbers['n'] , 0, quantum_numbers['m'])   

    wavefunction_210 = R_nl_210*f_lm_210*g_m_210
    wavefunction_200 = R_nl_200*f_lm_200*g_m_200
    wavefunction_minus = (1/np.sqrt(2)) * (wavefunction_200 + wavefunction_210)
    wavefunction_plus = (1/np.sqrt(2)) * (wavefunction_200 - wavefunction_210)

    p_minus = wavefunction_minus**2
    p_plus = wavefunction_plus**2
    # ----------------------

    fig = plt.figure()
    from matplotlib import cm
    plt.imshow(p_minus,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot) 
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Phi_minus probability distribution')
    plt.show()


def main(config_path=None):

    if config_path is None:
        root = "/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/"
        config_path = root + "experiment_configs/TISE_stark_effect/config_30index0.yaml"

    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    experiment_params = config["experiment_params"]
    network_params = config["network_params"]
    prior_params = config["prior_params"]
    results_path = config["results_path"]
    #results_path = '/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/results_TISE_stark_effect/' # path in local

    # To be specified quantum numbers are n and m, since the manifolds are defined only by m for a given n,
    # l has then the values: m <= l <= n-1
    quantum_numbers = experiment_params["quantum_numbers"]
    electric_field = float(experiment_params["electric_field"])
    index = experiment_params["index"] # which energy to consider for given manifold
    radial_extent = experiment_params["radial_extent"]

    n, m = quantum_numbers.values()
    results_path = results_path + str(n) + str(m) + "index" + str(index) + "/"
    import os
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    model, geom, data, net = create_model(experiment_params, network_params)


    # MAIN EXPERIMENTS: decreasing learning rate
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
    # # PRIOR CONVERGENCE TEST
    # prior_data = dde.data.PDE(geom, solution_prior_20index0, bcs=[], num_domain=1500, num_boundary=600)
    # prior_save_path = results_path + "prior/model"
    # compile_train_args = {'optimizer':'adam', 'lr':1e-3, 'epochs':10000}
    # model = CustomLossModel(data, net)
    # model.learn_prior(prior_data, prior_save_path, **compile_train_args)
    # model.compile("adam", lr=1.0e-5, loss='MSE')
    # #model.compile("adam", lr=1.0e-5, loss='NormalizationLoss')
    # ---------------------------------
    #model.train(epochs=10000) # TODO: uncomment if not using MAIN EXPERIMENTS


    # ## RESTORE BEST STEP ## TODO: uncomment when not using MAIN EXPERIMENTS
    # tf.compat.v1.reset_default_graph()
    # model, geom, data, net = create_model(pde_extra_arguments, index)
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

    eigenvalues, eigenvectors = TISE_stark_effect_exact(r, theta, phi, electric_field, n, m)
    p_gt = eigenvectors[index]**2 # Probability distribution

    r = r.reshape((n_points*n_points, 1))
    theta = theta.reshape((n_points*n_points, 1))
    phi = phi.reshape((n_points*n_points, 1))
    input_format = np.hstack((r, theta, phi))
    wavefunction_pred = model.predict(input_format)
    wavefunction_pred = wavefunction_pred[:,0:1] # real part
    # normalization
    normalization_constant = normalize_output(model=model, experiment_params=experiment_params)
    wavefunction_pred = normalization_constant * wavefunction_pred
    print("====== HERE ======")
    print(normalization_constant)
    ###############
    wavefunction_pred = wavefunction_pred.reshape((n_points, n_points))
    p_pred = wavefunction_pred**2

    from matplotlib import cm
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
    ener = eigenvalues[index] / (- 3 * e * electric_field * a0 / 2)
    fig.suptitle('{n},{m} MANIFOLD, ENERGY: {e}'.format(n=n, m=m, e=np.rint(ener)) + "$\Delta$E")
    plt.savefig(results_path + '{n}{m}index{i}.png'.format(n=n, m=m, i=index))



if __name__ == "__main__":

    #root = "/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/"
    root = "./"
    default_config_path = root + "experiment_configs/TISE_stark_effect/config_30index0.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, \
        default=default_config_path ,help='Path to the experiment config file.')
    args = parser.parse_args()
    config_path = args.config_path

    main(config_path)