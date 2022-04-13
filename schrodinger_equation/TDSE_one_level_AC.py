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
        self.results_path = config["results_path"]

    def solution_prior(self, args, wavefunction):
        t = args[:,0:1]
        u, v = wavefunction[:,0:1], wavefunction[:,1:2]

        F = self.experiment_params["F"]
        omega = self.experiment_params["omega"]
        d = self.experiment_params["d"]
        alpha = self.experiment_params["alpha"]
        u_gt, v_gt = TDSE_one_level_AC_exact(t, F, omega, d, alpha, backend="tensorflow")

        return [u-u_gt, v-v_gt]

    def fourier_prior_2d(self, args, wavefunction):
        t = args[:,0:1]
        u, v = wavefunction[:,0:1], wavefunction[:,1:2]
        basis = FourierBasis(max_k=10)
        representation = basis.compute(t)
        return [representation - u, representation - v]

    def pde(self, args, wavefunction, **pde_extra_arguments):
        """
        args = (t)
        wavefunction = (u, v), real and imaginary parts!
        """
        t = args[:,0:1]
        u, v = wavefunction[:,0:1], wavefunction[:,1:2]

        d = pde_extra_arguments["d"]
        F = pde_extra_arguments["F"]
        alpha = pde_extra_arguments["alpha"]
        omega = pde_extra_arguments["omega"]

        c = (-d*F*tf.cos(omega*t) - 0.5*alpha*F**2*tf.cos(omega*t)**2)
        du_dt = dde.grad.jacobian(wavefunction, args, i=0, j=0)
        dv_dt = dde.grad.jacobian(wavefunction, args, i=1, j=0)

        #return [du_dt - c*v, dv_dt + c*u]

        normalization = (u**2 + v**2) - 1
        return [du_dt - c*v, dv_dt + c*u, normalization]

    def create_model(self):

        pde_extra_arguments = self.experiment_params

        alpha = self.experiment_params["alpha"]
        F = self.experiment_params["F"]
        quasi_energy = - 0.25 * alpha * F**2
        period = np.abs(2*pi / quasi_energy)

        temporal_extent = self.experiment_params["temporal_extent"]
        geom = dde.geometry.Interval(0, period*temporal_extent)
        def boundary_left(args, on_boundary):
            return on_boundary and np.isclose(args[0], 0)
        def ic_func_u(args):
            return 1
        def ic_func_v(args):
            return 0
        ic_u = dde.DirichletBC(geom, ic_func_u, boundary_left, component=0) # We use Dirichlet BC as IC
        ic_v = dde.DirichletBC(geom, ic_func_v, boundary_left, component=1)

        num_domain = self.network_params["num_domain"]
        num_initial = self.network_params["num_initial"]
        data = CustomPDE(geom, self.pde, bcs=[ic_u, ic_v], num_domain=num_domain, num_boundary=num_initial, pde_extra_arguments=pde_extra_arguments)

        if self.network_params["backbone"] == "FNN":
            #net = dde.maps.FNN([3] + [50] * 4 + [2], "tanh", "Glorot normal")
            net = dde.maps.FNN(self.network_params["layers"], "tanh", "Glorot normal")
        elif self.network_params["backbone"] == "ResNet":
            input_size = self.network_params["input_size"]
            output_size = self.network_params["output_size"]
            num_neurons = self.network_params["num_neurons"]
            num_blocks = self.network_params["num_blocks"]
            net = dde.maps.ResNet(input_size, output_size, num_neurons, num_blocks, "tanh", "Glorot normal")
        elif self.network_params["backbone"] == "MsFFN":
            layers = self.network_params["layers"]
            sigmas = self.network_params["sigmas"]
            sigmas = [sigma/period for sigma in sigmas] # deviation chosen as a function of the period
            net = dde.maps.MsFFN(layers, "tanh", "Glorot normal", sigmas)
        else:
            raise NotImplementedError("Only FNN and ResNet backbones are experiment ready!")

        model = dde.Model(data, net)
        return model, geom, data, net

    def normalize_output(self):
        """
        This setting doesn't require normalization, since the normalization has been set using IC
        """
        pass

    def plot(self, t, a_gt, a_pred, save=True):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,7))
        ax1.title.set_text('Real part')
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Amplitude")
        ax1.plot(t, a_gt[0], "-b", label="Analytical solution", linewidth=3)
        ax1.plot(t, a_pred[:,0], "--r", label="PINN prediction", linewidth=3)

        ax2.title.set_text('Imaginary part')
        ax2.set_xlabel("Time [s]")
        ax2.plot(t, a_gt[1], "-b", label="Analytical solution", linewidth=3)
        ax2.plot(t, a_pred[:,1], "--r", label="PINN prediction", linewidth=3)
        ax2.legend(loc="upper right")
        if save:
            plt.savefig(self.results_path + "result.png")
        else:
            plt.show()
    
    def main(self):
        import os
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        model, geom, data, net = self.create_model()

        # ---------------------------------
        # PRIOR LEARING # TODO: add prior params
        prior_data = CustomPDE(geom, self.fourier_prior_2d, bcs=[], \
            num_domain=500, num_boundary=200)
        prior_save_path = self.results_path + "210_prior/model"
        compile_train_args = {'optimizer':'adam', 'lr':1e-3, 'epochs':10000}
        model = CustomLossModel(data, net)
        model.learn_prior(prior_data, prior_save_path, **compile_train_args)
        # ---------------------------------
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
        # ---------------------------------
        # # PRIOR CONVERGENCE TEST
        # prior_data = CustomPDE(geom, self.solution_prior, bcs=[], \
        #     num_domain=500, num_boundary=200)
        # prior_save_path = self.results_path + "prior/model"
        # compile_train_args = {'optimizer':'adam', 'lr':1e-3, 'epochs':10000}
        # model = CustomLossModel(data, net)
        # model.learn_prior(prior_data, prior_save_path, **compile_train_args)
        # model.compile("adam", lr=1.0e-3, loss='MSE')
        # model.train(epochs=10000) # TODO: uncomment when NOT running MAIN EXPERIMENTS
        # ---------------------------------

        t = geom.uniform_points(10000)
        F = self.experiment_params["F"]
        omega = self.experiment_params["omega"]
        d = self.experiment_params["d"]
        alpha = self.experiment_params["alpha"]
        a_gt = TDSE_one_level_AC_exact(t, F, omega, d, alpha)

        t_input = t.reshape(-1,1)
        a_pred = model.predict(t_input)

        self.plot(t, a_gt, a_pred)


if __name__ == "__main__":

    root = "/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/"
    #root = "./"
    default_config_path = root + "experiment_configs/TDSE_one_level_AC/config.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, \
        default=default_config_path ,help='Path to the experiment config file.')
    args = parser.parse_args()
    config_path = args.config_path

    solver = Solver(config_path)
    solver.main()
