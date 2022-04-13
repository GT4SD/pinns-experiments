from os import supports_bytes_environ
from deepxde.gradients import jacobian
from numpy.core.fromnumeric import _all_dispatcher
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
from pinnse.schrodinger_eq_exact_values import TISE_hydrogen_exact, TISE_stark_effect_exact, TDSE_one_level_AC_exact, normalization
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

class Solver():

    def __init__(self, config_path):
        super().__init__()
        with open(config_path, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.experiment_params = config["experiment_params"]
        self.network_params = config["network_params"]
        self.prior_params = config["prior_params"]
        self.SCID_TDSE_paths = config["SCID_TDSE_paths"]
        self.results_path = config["results_path"]

    def compute_primitive_conversion_factor(self, method="integration"):
        """
        method in ["integration", "simple"]
        """
        assert method in ["integration", "simple"], "method can either be integration or simple!"
        initial_state = self.experiment_params["initial_state"]
        n, l, m = initial_state.values()
        init_wfn_path = self.SCID_TDSE_paths["initial_wf_path"]
        
        # TODO: currently only implemented for 1s-2p --> generalize!
        with open(init_wfn_path+"/H-1S-2Pz-WU-L000-M+0000", "r") as f:
            wavefunction = f.readlines()
        radius = np.array([float(x.split()[1]) for x in wavefunction[1:]])
        right_wfn_re = np.array([float(x.split()[4]) for x in wavefunction[1:]])
        if method=="integration":
            factor = np.sqrt(1/simps(right_wfn_re**2, radius))
        elif method=="simple":
            gt_, _, _ = TISE_hydrogen_exact(radius, 0, 0, n, l, m)
            factor = gt_[0] / (right_wfn_re/radius)[0]
        return factor

    def compute_conversion_factor(self, timestep_vals_conv_factor, radius, method="integration"):
        assert method == "integration", "Only possible method is integration!"
        if method == "integration":
            wf = 0
            for k, val in enumerate(timestep_vals_conv_factor):
                th = np.linspace(0, np.pi, len(radius))
                ph = np.linspace(0, 2*np.pi, len(radius))
                R, Theta, Phi = np.meshgrid(radius, th, ph)

                _, f_simul, g_simul = TISE_hydrogen_exact(0, th, ph, k+1, k, 0)
                R_simul, f_simul, g_simul = np.meshgrid(val, f_simul, g_simul)
                wf_simul = R_simul*f_simul*g_simul

                wf = wf+wf_simul
            integrand = wf**2 * R**2 * np.sin(Theta)
            integral = simps(integrand, radius)
            integral = simps(integral, th)
            integral = simps(integral, ph)
            conversion_factor = np.sqrt(1/integral)
        return conversion_factor

    def data_loader(self):
        """
        Ground truth data loader from SCID-TDSE runs
        """
        data_loader_params = self.experiment_params["data_loader"]
        dt = data_loader_params["dt"]
        timesteps = data_loader_params["timesteps"]
        angular_momentum_channels = data_loader_params["angular_momentum_channels"]
        num_data = data_loader_params["num_data"]
        conversion = data_loader_params["conversion"]
        # We discard initial_wfn and final_wfn, since intermediate_wfn contains all needed information
        wfn_path = self.SCID_TDSE_paths["intermediate_wf_path"]

        input_data_list, solution_data_list = [], []
        for timestep in range(timesteps):
            # print("We are in timestep ", timestep)
            if data_loader_params["num_theta"] == "boundary":
                theta_ = np.array([0, pi])
            else:
                theta_ = np.linspace(0,pi, data_loader_params["num_theta"])
            if data_loader_params["num_phi"] == "boundary":
                phi_ = np.array([0, 2*pi])
            else:
                phi_ = np.linspace(0,pi, data_loader_params["num_phi"])

            timestep_vals = []
            timestep_vals_conv_factor = []
            radius_loaded = False
            for channel in angular_momentum_channels:
                # print("--> channel: ", channel)
                file = "/" + str(int(timestep)) + "-L" + f"{int(channel):03}" + "-M+0000"
                with open(wfn_path+file, "r") as f:
                    wavefunction = f.readlines()
                if not radius_loaded:
                    radius_ = np.array([float(x.split()[1]) for x in wavefunction[1:]])
                    radius = np.repeat(radius_, len(theta_)*len(phi_)).reshape(-1,1)
                    radius_loaded = True
                left_wfn_re = np.array([float(x.split()[2]) for x in wavefunction[1:]])
                left_wfn_im = np.array([float(x.split()[3]) for x in wavefunction[1:]])
                right_wfn_re_ = np.array([float(x.split()[4]) for x in wavefunction[1:]])
                right_wfn_im = np.array([float(x.split()[5]) for x in wavefunction[1:]])

                right_wfn_re = np.repeat(right_wfn_re_, len(theta_)*len(phi_)).reshape(-1,1)
                if conversion == "primitive":
                    primitive_conversion_factor = self.compute_primitive_conversion_factor()
                    timestep_vals.append(primitive_conversion_factor * (right_wfn_re/radius))
                elif conversion == "complex":
                    timestep_vals.append(right_wfn_re/radius)
                    timestep_vals_conv_factor.append(right_wfn_re_ / radius_)
                else:
                    timestep_vals.append(right_wfn_re/radius)

            theta = np.repeat(theta_, len(phi_))
            theta = np.tile(theta, len(radius_)).reshape(-1,1)
            phi = np.tile(phi_, len(radius_)*len(theta_)).reshape(-1,1)
            time = timestep*dt* np.ones_like(radius)
            input_data = np.hstack([radius, theta, phi, time])

            for i, channel in enumerate(angular_momentum_channels):
                R_gt, f_gt, g_gt = TISE_hydrogen_exact(radius, theta, phi, channel+1, channel, 0)
                timestep_vals[i] *= f_gt*g_gt
            timestep_val = np.sum(timestep_vals, axis=0)

            #####################################
            # Computation of conversion factor on every step takes a lot of time! --> try without conversion factor
            if conversion == "complex":
                conversion_factor = self.compute_conversion_factor(timestep_vals_conv_factor, radius_)
                timestep_val *= conversion_factor
            #####################################

            input_data_list.append(input_data)
            solution_data_list.append(timestep_val)

        supervised_input = np.vstack(input_data_list)
        supervised_output = np.vstack(solution_data_list)

        num_rows = supervised_input.shape[0]
        random_indices = np.random.choice(num_rows, size=num_data, replace=False)
        supervised_input = supervised_input[random_indices, :]
        supervised_output = supervised_output[random_indices, :]

        return supervised_input, supervised_output

    def alpha_from_width(self, omega, width, tol=1e-40):
        """
        Method directly copied from SCID_TDSE library
        """
        def iteration(alpha):
            return (2/width**2) * (np.log(2) + np.log(1 + (width*alpha/omega)**2))
        alpha0 = iteration(0)
        alpha_iter = iteration(alpha0)
        alpha = alpha0
        while np.abs(alpha - alpha_iter) > tol:
            alpha = alpha_iter
            alpha_iter = iteration(alpha_iter)
        return alpha
        
    def zGaussianVP(self, t, vp_scale, vp_params, backend="tensorflow"):
        assert backend in ["tensorflow", "numpy"], "Only tensorflow & numpy backends are available!"
        omega, phase, t0, width, t1, t2 = vp_params
        alpha = self.alpha_from_width(omega, width)
        tau = t - t0
        if backend=="tensorflow":
            t_pulse = t1 + (2/pi) * (t2-t1) * tf.tan((pi/2)*(tf.abs(tau) - t1)/(t2-t1))
            f = tf.where(tf.abs(tau)<=t1, tf.math.exp(-alpha*tau**2), tf.math.exp(-alpha*t_pulse**2))
            f = tf.where(tf.abs(tau)<=t2, f, tf.zeros_like(f))
            return vp_scale * tf.cos(omega*tau + phase) * f
        elif backend=="numpy":
            t_pulse = t1 + (2/pi) * (t2-t1) * np.tan((pi/2)*(np.abs(tau) - t1)/(t2-t1))
            f = np.where(np.abs(tau)<=t1, np.exp(-alpha*tau**2), np.exp(-alpha*t_pulse**2))
            f = np.where(np.abs(tau)<=t2, f, np.zeros_like(f))
            return vp_scale * np.cos(omega*tau + phase) * f

    def pde_polar(self, args, wavefunction, **pde_extra_arguments):
        ''' Everything in atomic units
        args = (r, theta, phi, t)
        wavefunction = (u, v), real and imaginary parts!
        '''
        initial_state = pde_extra_arguments["initial_state"]
        n,l,m = initial_state.values() # will not use these initial state values, since the atom will not remain in the initial state!
        vp = pde_extra_arguments["vp"]
        vp_shape, vp_scale, vp_params = vp.values()
        vp_scale = float(vp_scale)

        # constants
        Z = 1
        A = l * (l+1)
        E_n = - (electron_mass* Z**2 * e**4) / (8 * epsilon_0**2 * h**2 * n**2)

        r, theta, phi, t = args[:,0:1], args[:,1:2], args[:,2:3], args[:,3:4]
        u, v = wavefunction[:,0:1], wavefunction[:,1:2]
        if vp_shape == "z Gaussian":
            A_z = self.zGaussianVP(t, vp_scale, vp_params, backend="tensorflow")
        else:
            # TODO: implement
            A_z = 0

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

        # TDSE
        c1 = - (hbar**2 * r**2 * tf.sin(theta)**2 / (2*electron_mass)) * du_drr
        c2 = - (hbar**2 * r * tf.sin(theta)**2 / (electron_mass)) * du_dr
        cL2 = (-hbar**2/(2*electron_mass)) * (tf.sin(theta)*tf.cos(theta)*du_dtheta + tf.sin(theta)**2*du_dthetatheta + du_dphiphi)
        c3 = (- Z *e**2 * r *tf.sin(theta)**2 / (4*pi*epsilon_0)) * u
        cA1 = - (e*hbar/electron_mass) * A_z * (r**2*tf.sin(theta)**2*tf.cos(theta)*dv_dr - r*tf.sin(theta)**3*dv_dtheta)
        cA2 = (e**2/(2*electron_mass)) * A_z**2 *r**2*tf.sin(theta)**2 * u
        ct =  hbar*r**2*tf.sin(theta)**2 * v
        ex1 = c1+c2+cL2+c3+cA1+cA2+ct

        c1v = - (hbar**2 * r**2 * tf.sin(theta)**2 / (2*electron_mass)) * dv_drr
        c2v = - (hbar**2 * r * tf.sin(theta)**2 / (electron_mass)) * dv_dr
        cL2v = (-hbar**2/(2*electron_mass)) * (tf.sin(theta)*tf.cos(theta)*dv_dtheta + tf.sin(theta)**2*dv_dthetatheta + dv_dphiphi)
        c3v = (- Z *e**2 * r *tf.sin(theta)**2 / (4*pi*epsilon_0)) * v
        cA1v = (e*hbar/electron_mass) * A_z * (r**2*tf.sin(theta)**2*tf.cos(theta)*du_dr - r*tf.sin(theta)**3*du_dtheta)
        cA2v = (e**2/(2*electron_mass)) * A_z**2 *r**2*tf.sin(theta)**2 * v
        ctv =  - hbar*r**2*tf.sin(theta)**2 * u
        ex2 = c1v+c2v+cL2v+c3v+cA1v+cA2v+ctv

        return [ex1, ex2]
    
    def TDSE_unperturbed_test(self):
        # TODO
        """
        Testing of the pde using unperturbed orbital evolution
        """
        pass

    def create_model(self):
        initial_state = self.experiment_params["initial_state"]
        radial_extent = self.experiment_params["radial_extent"]
        temporal_extent = self.experiment_params["temporal_extent"]
        vp = self.experiment_params["vp"]

        n,l,m = initial_state.values()

        pde_extra_arguments = {"initial_state": initial_state, \
            "vp": vp}
        # ---------------------------------
        # geom = dde.geometry.Cuboid(xmin=[0,0,0], xmax=[30*a0, np.pi, 2*np.pi])
        # def boundary_right(x, on_boundary):
        #     return on_boundary and np.isclose(x[0], 30*a0, atol=a0*1e-08)
        geom = dde.geometry.Cuboid(xmin=[0,0,0], xmax=[radial_extent, np.pi, 2*np.pi])
        timedomain = dde.geometry.TimeDomain(0,temporal_extent)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        def boundary_right(args, on_boundary):
            return on_boundary and np.isclose(args[0], radial_extent)
        def g_boundary_left(args, on_boundary):
            return on_boundary and np.isclose(args[2], 0)
        def ic_func_u(args):
            r, theta, phi = args[:,0:1], args[:,1:2], args[:,2:3]
            n, l, m = initial_state.values()
            R, f, g = TISE_hydrogen_exact(r, theta, phi, n,l,m)
            return R*f*g
        def ic_func_v(args):
            r, theta, phi = args[:,0:1], args[:,1:2], args[:,2:3]
            n, l, m = initial_state.values()
            R, f, g = TISE_hydrogen_exact(r, theta, phi, n,l,m)
            g = np.sin(m*phi) / np.sqrt(2*pi)
            return R*f*g
        bc_u = dde.DirichletBC(geomtime, lambda x:0, boundary_right, component=0) # Wavefunction assumed to go 0 on the domain boundaries, which might not be true!
        bc_v = dde.DirichletBC(geomtime, lambda x:0, boundary_right, component=1)
        bc_g_u = PeriodicBC(geomtime, 2, g_boundary_left, periodicity="symmetric", component=0)
        bc_g_v = PeriodicBC(geomtime, 2, g_boundary_left, periodicity="symmetric", component=1)
        ic_u = dde.IC(geomtime, ic_func_u, lambda _, on_initial: on_initial)
        ic_v = dde.IC(geomtime, ic_func_v, lambda _, on_initial: on_initial, component=1)
        if self.network_params["strict_boundary"]:
            supervised_inp, supervised_outp = self.data_loader()
            # TODO: there is too much supervised data! Add a percentage param to reduce the amount of supervised data
            strict_bc_u = dde.PointSetBC(supervised_inp, supervised_outp, component=0) # supervised data for real part of the wf
            data = CustomTimePDE(geomtime, self.pde_polar, ic_bcs=[bc_u, bc_v, bc_g_u, bc_g_v, ic_u, ic_v, strict_bc_u], \
                num_domain=self.network_params["num_domain"], num_boundary=self.network_params["num_domain"], \
                    num_initial=self.network_params["num_initial"], pde_extra_arguments=pde_extra_arguments)
        else:
            data = CustomTimePDE(geomtime, self.pde_polar, ic_bcs=[bc_u, bc_v, bc_g_u, bc_g_v, ic_u, ic_v], \
                num_domain=self.network_params["num_domain"], num_boundary=self.network_params["num_domain"], \
                    num_initial=self.network_params["num_initial"], pde_extra_arguments=pde_extra_arguments)

        if self.network_params["backbone"] == "FNN":
            #net = dde.maps.FNN([3] + [50] * 4 + [2], "tanh", "Glorot normal")
            net = dde.maps.FNN(self.network_params["layers"], "tanh", "Glorot normal")
        elif self.network_params["backbone"] == "ResNet":
            input_size = self.network_params["input_size"]
            output_size = self.network_params["output_size"]
            num_neurons = self.network_params["num_neurons"]
            num_blocks = self.network_params["num_blocks"]
            net = dde.maps.ResNet(input_size, output_size, num_neurons, num_blocks, "tanh", "Glorot normal")
        else:
            raise NotImplementedError("Only FNN and ResNet backbones are experiment ready!")

        model = dde.Model(data, net)
        return model, geomtime, data, net

    def normalize_output(self, model):
        """
        Normalization is currently forced using PointSetBC
        --> in case of inconsistencies, steal the implementation from TDSE_unperturbed
        """
        pass

    def plot(self, rmax, p_gt, p_pred, timestep, save=True):
        # TODO: need to decide on what to plot, prob distribution will be indistinguishable to eye
        pass

    def main(self):
        self.results_path = self.results_path
        import os
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        model, geomtime, data, net = self.create_model()

        # MAIN EXPERIMENTS: decreasing learning rate
        # TODO: comment out model restoring when running on cluster
        assert len(self.network_params["lrs"]) == len(self.network_params["optimizers"]) and \
            len(self.network_params["lrs"]) == len(self.network_params["epochs"]), "Incompatible network parameter lengths!"
        for i in range(len(self.network_params["lrs"])):
            lr = float(self.network_params["lrs"][i])
            optimizer = self.network_params["optimizers"][i]
            epoch = self.network_params["epochs"][i]
            model.compile(optimizer=optimizer, lr=lr, loss="MSE", loss_weights=self.network_params["loss_weights"])
            # with open(results_path+"model/best_step.txt") as f: # TODO: add this after first loop
            #     best = f.readlines()[0]
            # model.restore(results_path + "model/model.ckpt-" + best, verbose=1)
            checker = dde.callbacks.ModelCheckpoint(
                self.results_path+"model/model.ckpt", save_better_only=True, period=1000
            )
            losshistory, train_state = model.train(epochs=epoch, callbacks=[checker])
            with open(self.results_path+"model/best_step.txt", "w") as text_file:
                text_file.write(str(train_state.best_step))

        # TODO: finish-up

if __name__ == "__main__":

    root = "/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/"
    #root = "./"
    default_config_path = root + "experiment_configs/TDSE_laser_field/config_1s2p.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, \
        default=default_config_path ,help='Path to the experiment config file.')
    args = parser.parse_args()
    config_path = args.config_path

    solver = Solver(config_path)
    solver.main()

    # a, b = solver.data_loader()
    # print(a.shape)

