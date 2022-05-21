import sys
import yaml

import deepxde as dde
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass, physical_constants

from .schrodinger_eq_exact_values import (
    TISE_hydrogen_exact,
    TDSE_unperturbed_exact,
    TISE_stark_effect_exact,
    TDSE_one_level_AC_exact,
)
from .utils import (
    count_zero_in_decimal_number,
    hbar,
    a0,
    e,
    electron_mass,
    epsilon_0,
    h,
)


class EquationSystemTest(object):
    def __init__(self):
        super().__init__()
        pass

    def pde(self, x, y):
        u = y[:, 0:1]  # real part
        v = y[:, 1:2]  # imaginary part

        du_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dv_dx = dde.grad.jacobian(y, x, i=1, j=0)

        du_dxx = dde.grad.hessian(y, x, i=0, j=0, component=0)
        dv_dxx = dde.grad.hessian(y, x, i=0, j=0, component=1)

        # return [du_dx + v, dv_dx - u]
        return [du_dxx + u, dv_dxx + v, du_dx + v, dv_dx - u]

    def ic_func_u(self, x):
        input = x[:, 0:1]
        return 0.5 * np.cos(input)

    def ic_func_v(self, x):
        input = x[:, 0:1]
        return 0.5 * np.sin(input)

    def solve(self):
        timedomain = dde.geometry.Interval(0, 2 * pi)

        def on_initial(x, on_boundary):
            return on_boundary and np.isclose(x[0], 0)

        # ic_u = dde.DirichletBC(timedomain, self.ic_func_u, on_initial, component=0)
        # ic_v = dde.DirichletBC(timedomain, self.ic_func_v, on_initial, component=1)
        ic_u = dde.DirichletBC(
            timedomain, self.ic_func_u, lambda _, on_boundary: on_boundary, component=0
        )
        ic_v = dde.DirichletBC(
            timedomain, self.ic_func_v, lambda _, on_boundary: on_boundary, component=1
        )

        data = dde.data.PDE(
            timedomain, self.pde, [ic_u, ic_v], num_domain=500, num_boundary=200
        )

        net = dde.maps.FNN([1] + [32] * 3 + [2], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=1.0e-3)
        model.train(10000)

        # === PLOT ===
        X = timedomain.uniform_points(10000)
        predictions = model.predict(X)

        plt.figure(figsize=(12, 8))
        plt.plot(X.reshape(-1), 0.5 * np.cos(X.reshape(-1)), ".k")
        plt.plot(X.reshape(-1), predictions[:, 0:1], ".r")
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(X.reshape(-1), 0.5 * np.sin(X.reshape(-1)), ".k")
        plt.plot(X.reshape(-1), predictions[:, 1:2], ".r")
        plt.show()


class Metrics(object):
    def __init__(self, gt, predicted):
        super().__init__()
        self.gt = gt
        self.predicted = predicted

    def relative_error(self):
        err = np.abs(self.predicted - self.gt)
        return np.mean(err) / np.mean(np.abs(self.gt))

    def absolute_error(self):
        err = self.predicted - self.gt
        return np.mean(np.abs(err))

    def wf_overlap(self):  # TODO --> implement this in physical_quantities.py
        pass


class PINNTest(object):
    def __init__(self, script=None, save_pth=None, config_path=None) -> None:
        super().__init__()
        assert script in [
            "TISE_hydrogen",
            "TISE_hydrogen_without_decomposition",
            "TISE_stark_effect",
            "TDSE_unperturbed",
            "TDSE_one_level_AC",
            "TDSE_laser_field",
        ], "Experiment script non-existent!"
        self.script = script
        self.save_pth = save_pth
        self.config_path = config_path  # in case of TDSE_one_level_AC
        with open(config_path, "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        self.experiment_params = self.config[
            "experiment_params"
        ]  # params like quantum_numbers, electric field etc.
        self.network_params = self.config["network_params"]
        self.results_path = self.config["results_path"]

    def _model_setup(self):
        sys.path.append(
            "/Users/lat/Desktop/Code/pinns-experiments/schrodinger_equation/"
        )

        if self.script == "TDSE_unperturbed":
            from TDSE_unperturbed import Solver

            self.solver = Solver(self.config_path)
            model, geom, data, net = self.solver.create_model()

        elif self.script == "TISE_hydrogen_without_decomposition":
            from TISE_hydrogen_without_decomposition import Solver

            self.solver = Solver(self.config_path)
            model, geom, data, net = self.solver.create_model()

        elif self.script == "TISE_stark_effect":
            from TISE_stark_effect import Solver

            self.solver = Solver(self.config_path)
            model, geom, data, net = self.solver.create_model()

        elif self.script == "TISE_hydrogen":
            return None, None, None, None

        elif self.script == "TDSE_one_level_AC":
            from TDSE_one_level_AC import Solver

            self.solver = Solver(self.config_path)
            model, geom, data, net = self.solver.create_model()

        elif self.script == "TDSE_laser_field":
            from TDSE_laser_field import Solver

            self.solver = Solver(self.config_path)
            model, geom, data, net = self.solver.create_model()

        model.compile("adam", lr=1e-3, loss="MSE")

        with open(self.save_pth + "best_step.txt") as f:
            best = f.readlines()[0]
        # TODO: solve this for good!
        try:
            model.restore(self.save_pth + "model.ckpt-" + best + ".ckpt", verbose=1)
        except:
            model.restore(self.save_pth + "model.ckpt-" + best, verbose=1)
        return model, geom, data, net

    def evaluate_metrics(self, sampled_points=10000):
        model, geom, data, net = self._model_setup()

        # X_test = data.train_points()
        if model is not None:
            X_test = geom.random_points(sampled_points)
        # X_test = geom.uniform_points(sampled_points)
        # TODO: uniform point sampling from domain takes interval width into consideration!:
        # use data.train_points() instead --> random sampling

        if self.script == "TDSE_unperturbed":
            r, theta, phi, t = X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3]
            quantum_numbers = self.experiment_params["quantum_numbers"]
            n, l, m = quantum_numbers.values()

            # # for the evaluations after the defined period
            # E_n = - (electron_mass * e**4) / (8 * epsilon_0**2 * h**2 * n**2)
            # period = np.abs(2*pi*hbar / E_n)
            # t += period
            # #########################################

            gt_real, gt_imag = TDSE_unperturbed_exact(r, theta, phi, t, n=n, l=l, m=m)
            gt_real, gt_imag = gt_real.reshape((-1, 1)), gt_imag.reshape((-1, 1))

            predicted = model.predict(X_test)
            predicted_real, predicted_imag = predicted[:, 0:1], predicted[:, 1:2]

            evaluator_real = Metrics(gt_real, predicted_real)
            evaluator_imag = Metrics(gt_imag, predicted_imag)
            gt = gt_real**2 + gt_imag**2
            pred = predicted_real**2 + predicted_imag**2
            evaluator_tot = Metrics(gt, pred)

            print("=== EVALUATION RESULTS ===")
            print("Relative Error of Real Part: ", evaluator_real.relative_error())
            print("Absolute Error of Real Part: ", evaluator_real.absolute_error())
            print("--------------------------")
            print("Relative Error of Imaginary Part: ", evaluator_imag.relative_error())
            print("Absolute Error of Imaginary Part: ", evaluator_imag.absolute_error())
            print("--------------------------")
            print(
                "Relative Error of Complete Prob. Distribution: ",
                evaluator_tot.relative_error(),
            )
            print(
                "Absolute Error of Complete Prob. Distribution: ",
                evaluator_tot.absolute_error(),
            )

        elif self.script == "TISE_hydrogen_without_decomposition":
            r, theta, phi = X_test[:, 0], X_test[:, 1], X_test[:, 2]
            quantum_numbers = self.experiment_params["quantum_numbers"]
            n, l, m = quantum_numbers.values()

            gt = TISE_hydrogen_exact(r, theta, phi, n=n, l=l, m=m)
            gt = gt[0] * gt[1] * gt[2]
            gt = gt.reshape((-1, 1)) ** 2

            predicted_u, predicted_v = (
                model.predict(X_test)[:, 0:1],
                model.predict(X_test)[:, 1:2],
            )  # real part
            ## NORMALIZATION ##
            normalization_constant = self.solver.normalize_output(model=model)
            predicted = (normalization_constant * predicted_u) ** 2 + (
                normalization_constant * predicted_v
            ) ** 2
            ####################

            evaluator = Metrics(gt, predicted)

            print("=== EVALUATION RESULTS ===")
            print("Relative Error: ", evaluator.relative_error())
            print("Absolute Error: ", evaluator.absolute_error())

        elif self.script == "TISE_stark_effect":
            r, theta, phi = X_test[:, 0], X_test[:, 1], X_test[:, 2]
            quantum_numbers = self.experiment_params["quantum_numbers"]
            electric_field = float(self.experiment_params["electric_field"])
            index = self.experiment_params["index"]
            n, m = quantum_numbers.values()

            _, gt = TISE_stark_effect_exact(r, theta, phi, electric_field, n, m)
            gt = gt[index].reshape((-1, 1)) ** 2

            predicted_u, predicted_v = (
                model.predict(X_test)[:, 0:1],
                model.predict(X_test)[:, 1:2],
            )  # real part
            ## NORMALIZATION ##
            normalization_constant = self.solver.normalize_output(model=model)
            predicted = (normalization_constant * predicted_u) ** 2 + (
                normalization_constant * predicted_v
            ) ** 2
            ####################

            evaluator = Metrics(gt, predicted)

            print("=== EVALUATION RESULTS ===")
            print("Relative Error: ", evaluator.relative_error())
            print("Absolute Error: ", evaluator.absolute_error())

        elif self.script == "TISE_hydrogen":
            from TISE_hydrogen import (
                create_model_R_nl,
                create_model_f_lm,
                create_model_g_m,
            )

            R_params = self.network_params["R"]
            f_params = self.network_params["f"]
            g_params = self.network_params["g"]

            quantum_numbers = self.experiment_params["quantum_numbers"]
            n, l, m = quantum_numbers.values()

            # R MODEL
            R_model, geom, data, net = create_model_R_nl(
                self.experiment_params, R_params
            )
            R_model.compile("adam", lr=1e-3, loss="MSE")
            with open(self.save_pth + "model_R/" + "best_step.txt") as f:
                best = f.readlines()[0]
            R_model.restore(
                self.save_pth + "model_R/" + "model.ckpt-" + best, verbose=1
            )

            # normalization
            X = geom.uniform_points(1000)
            r_norm = X.reshape(-1)
            R_nl_norm = R_model.predict(X)
            R_nl_norm = R_nl_norm.reshape(-1)
            integrand = R_nl_norm**2 * r_norm**2
            C = simps(integrand, r_norm)
            normalization_constant_R = 1.0 / np.sqrt(C)

            r_metric = geom.random_points(sampled_points)
            R_nl_pred = R_model.predict(r_metric)
            R_nl_pred = R_nl_pred * normalization_constant_R
            tf.compat.v1.reset_default_graph()

            # f MODEL
            if f_params["prior"] == "Fourier":
                (
                    f_model,
                    geom,
                    data,
                    net,
                    prior_data,
                    prior_save_path,
                    compile_train_args,
                ) = create_model_f_lm(
                    self.experiment_params, f_params, self.results_path
                )
            else:
                f_model, geom, data, net = create_model_f_lm(
                    self.experiment_params, f_params, self.results_path
                )
            f_model.compile("adam", lr=1e-3, loss="MSE")
            with open(self.save_pth + "model_f/" + "best_step.txt") as f:
                best = f.readlines()[0]
            f_model.restore(
                self.save_pth + "model_f/" + "model.ckpt-" + best, verbose=1
            )

            # normalization
            X = geom.uniform_points(1000)
            theta_norm = X.reshape(-1)
            f_lm_norm = f_model.predict(X)
            f_lm_norm = f_lm_norm.reshape(-1)
            integrand = f_lm_norm**2 * np.sin(theta_norm)
            C = simps(integrand, theta_norm)
            normalization_constant_f = 1.0 / np.sqrt(C)

            theta_metric = geom.random_points(sampled_points)
            f_lm_pred = f_model.predict(theta_metric)
            f_lm_pred = f_lm_pred * normalization_constant_f
            tf.compat.v1.reset_default_graph()

            # g MODEL
            g_model, geom, data, net = create_model_g_m(
                self.experiment_params, g_params
            )
            g_model.compile("adam", lr=1e-3, loss="MSE")
            with open(self.save_pth + "model_g/" + "best_step.txt") as f:
                best = f.readlines()[0]
            g_model.restore(
                self.save_pth + "model_g/" + "model.ckpt-" + best, verbose=1
            )

            # normalization
            X = geom.uniform_points(1000)
            phi_norm = X.reshape(-1)
            g_m_norm = g_model.predict(X)
            g_m_norm_real, g_m_norm_imag = g_m_norm[:, 0], g_m_norm[:, 1]
            integrand = g_m_norm_real**2 + g_m_norm_imag**2
            C = simps(integrand, phi_norm)
            normalization_constant_g = 1.0 / np.sqrt(C)

            phi_metric = geom.random_points(sampled_points)
            g_m_pred = g_model.predict(phi_metric)
            g_m_pred = g_m_pred * normalization_constant_g
            tf.compat.v1.reset_default_graph()
            #####################

            R_nl_gt, f_lm_gt, g_m_gt = TISE_hydrogen_exact(
                r_metric.reshape(-1),
                theta_metric.reshape(-1),
                phi_metric.reshape(-1),
                n,
                l,
                m,
            )
            wavefunction_gt = R_nl_gt * f_lm_gt * g_m_gt
            gt = wavefunction_gt**2  # Probability distribution

            wavefunction_pred = (
                R_nl_pred * f_lm_pred * g_m_pred[:, 0:1]
            )  # take the real part
            wavefunction_pred = wavefunction_pred.reshape(-1)
            predicted = wavefunction_pred**2

            evaluator = Metrics(gt, predicted)

            print("=== EVALUATION RESULTS ===")
            print("Relative Error: ", evaluator.relative_error())
            print("Absolute Error: ", evaluator.absolute_error())

        elif self.script == "TDSE_one_level_AC":
            t = X_test[:, 0]
            F = self.experiment_params["F"]
            omega = self.experiment_params["omega"]
            d = self.experiment_params["d"]
            alpha = self.experiment_params["alpha"]

            # # for generalization evaluation
            # quasi_energy = - 0.25 * alpha * F**2
            # period = np.abs(2*pi / quasi_energy)
            # t += period
            # ###############################

            gt = TDSE_one_level_AC_exact(t, F, omega, d, alpha)
            t = t.reshape(-1, 1)
            pred = model.predict(t)

            predicted_real, predicted_imag = pred[:, 0], pred[:, 1]
            gt_real, gt_imag = gt

            evaluator_real = Metrics(gt_real, predicted_real)
            evaluator_imag = Metrics(gt_imag, predicted_imag)

            print("=== EVALUATION RESULTS ===")
            print("Relative Error of Real Part: ", evaluator_real.relative_error())
            print("Absolute Error of Real Part: ", evaluator_real.absolute_error())
            print("--------------------------")
            print("Relative Error of Imaginary Part: ", evaluator_imag.relative_error())
            print("Absolute Error of Imaginary Part: ", evaluator_imag.absolute_error())

        elif self.script == "TDSE_laser_field":
            (
                supervised_input,
                supervised_output,
                supervised_output_im,
            ) = self.solver.data_loader()

            predicted = model.predict(supervised_input)
            predicted_real, predicted_imag = predicted[:, 0:1], predicted[:, 1:2]

            evaluator_real = Metrics(supervised_output, predicted_real)
            evaluator_imag = Metrics(supervised_output_im, predicted_imag)

            print("=== EVALUATION RESULTS ===")
            print("Relative Error of Real Part: ", evaluator_real.relative_error())
            print("Absolute Error of Real Part: ", evaluator_real.absolute_error())
            print("--------------------------")
            print("Relative Error of Imaginary Part: ", evaluator_imag.relative_error())
            print("Absolute Error of Imaginary Part: ", evaluator_imag.absolute_error())

        tf.compat.v1.reset_default_graph()

    def evaluate_weights(self, save=False, plot_name=None):
        root = "/Users/lat/Desktop/Code/pinns-experiments/analysis_plots/"
        model, geom, data, net = self._model_setup()
        counter = 0

        param_amount = 0
        for i, var in enumerate(tf.compat.v1.trainable_variables()):
            param_amount += len(var.eval(session=model.sess).flatten())
        print(param_amount)
        return

        if "resnet" in self.config_path:
            subplots = int((len(tf.compat.v1.trainable_variables()) / 4) + 1)
        else:
            subplots = int(len(tf.compat.v1.trainable_variables()) / 2)
        fig, axs = plt.subplots(1, subplots, tight_layout=True)
        counter = 0
        plot_counter = 0
        for i, var in enumerate(tf.compat.v1.trainable_variables()):
            # We expect 1 weight matrix for dense layers, 2 weight matrices for residual layers
            if "resnet" in self.config_path:
                if len(var.eval(session=model.sess).shape) != 1:
                    counter += 1
                    plot = True
                    if (i == 0) or (i == len(tf.compat.v1.trainable_variables()) - 2):
                        to_plot = var.eval(session=model.sess).flatten()
                    elif counter % 2 == 0:
                        to_plot = var.eval(session=model.sess).flatten()
                        plot = False
                    elif counter % 2 == 1:
                        to_plot = np.hstack(
                            [to_plot, var.eval(session=model.sess).flatten()]
                        )
                    # print(var)
                    # print(var.eval(session=model.sess).shape)
                    # print(var.eval(session=model.sess).flatten())
                    if plot:
                        axs[plot_counter].hist(to_plot, bins=100, range=(-0.5, 0.5))
                        plot_counter += 1
            else:
                if len(var.eval(session=model.sess).shape) != 1:
                    axs[plot_counter].hist(
                        var.eval(session=model.sess).flatten(),
                        bins=100,
                        range=(-0.5, 0.5),
                    )
                    plot_counter += 1
            # print(var.eval(session=model.sess).shape)
        plt.show()

    def _plot(
        self, p_gt, p_pred, rmax=20, title=None, results_path=None, fig_name=None
    ):
        from matplotlib import cm

        vmax = max(np.max(p_gt), np.max(p_pred))
        exponent = count_zero_in_decimal_number(vmax)
        clb_title = str("1e-{e}".format(e=exponent))
        p_gt = p_gt * 10**exponent
        p_pred = p_pred * 10**exponent
        vmax = max(np.max(p_gt), np.max(p_pred))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        plt1 = ax1.imshow(
            p_gt,
            vmin=0,
            vmax=vmax,
            extent=[-rmax, rmax, -rmax, rmax],
            interpolation="none",
            origin="lower",
            cmap=cm.hot,
        )
        ax1.title.set_text("Exact solution")
        ax1.set_xlabel("x")
        ax1.set_ylabel("z")
        clb1 = fig.colorbar(plt1, ax=ax1, fraction=0.046, pad=0.04)
        clb1.ax.set_title(clb_title)

        plt2 = ax2.imshow(
            p_pred,
            vmin=0,
            vmax=vmax,
            extent=[-rmax, rmax, -rmax, rmax],
            interpolation="none",
            origin="lower",
            cmap=cm.hot,
        )
        ax2.title.set_text("PINN prediction")
        ax2.set_xlabel("x")
        clb2 = fig.colorbar(plt2, ax=ax2, fraction=0.046, pad=0.04)
        clb2.ax.set_title(clb_title)

        plt3 = ax3.imshow(
            np.abs(p_gt - p_pred),
            extent=[-rmax, rmax, -rmax, rmax],
            interpolation="none",
            origin="lower",
            cmap=cm.hot,
        )
        ax3.title.set_text("Absolute error")
        ax3.set_xlabel("x")
        clb2 = fig.colorbar(plt3, ax=ax3, fraction=0.046, pad=0.04)
        clb2.ax.set_title(clb_title)

        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=0.94)
        fig.subplots_adjust(left=0.06)
        if title is not None:
            fig.suptitle(title)
        if results_path == None:
            plt.show()
        else:
            plt.savefig(results_path + fig_name)

    def plots(self):
        model, geom, data, net = self._model_setup()

        ## PLOT FOR Y=0 ##
        rmax = 20
        n_points = 1000  # must be an even number
        x = np.linspace(-rmax, rmax, n_points)
        z = np.linspace(-rmax, rmax, n_points)
        X, Z = np.meshgrid(x, z)
        r = np.sqrt(X**2 + Z**2)
        theta = np.arctan(np.sqrt(X**2) / Z)
        theta = np.where(theta < 0, np.pi + theta, theta)
        phi = [
            pi * np.ones([n_points, int(np.floor(n_points / 2))]),
            np.zeros([n_points, int(np.ceil(n_points / 2))]),
        ]
        phi = np.hstack(phi)

        if self.script == "TDSE_unperturbed":
            quantum_numbers = self.experiment_params["quantum_numbers"]
            n, l, m = quantum_numbers.values()
            R_nl_gt, f_lm_gt, g_m_gt = TISE_hydrogen_exact(r, theta, phi, n, l, m)
            wavefunction_gt = R_nl_gt * f_lm_gt * g_m_gt

            r = r.reshape((n_points * n_points, 1))
            theta = theta.reshape((n_points * n_points, 1))
            phi = phi.reshape((n_points * n_points, 1))

            E_n = -(electron_mass * e**4) / (8 * epsilon_0**2 * h**2 * n**2)
            period = np.abs(2 * pi * hbar / E_n)
            time_points = [x * period / 4 for x in range(4)]

            # # for the evaluations after the defined period
            # time_points = [x+period for x in time_points]
            # #########################################

            for time_point in time_points:
                time = time_point * np.ones_like(r)
                input_format = np.hstack((r, theta, phi, time))
                wavefunction_pred = model.predict(input_format)
                wavefunction_pred_real = wavefunction_pred[:, 0:1]  # real part
                wavefunction_pred_imag = wavefunction_pred[:, 1:2]  # imaginary part
                # normalization
                normalization_constant = self.solver.normalize_output(model=model)
                wavefunction_pred_real = normalization_constant * wavefunction_pred_real
                wavefunction_pred_imag = normalization_constant * wavefunction_pred_imag
                ###############
                wavefunction_pred_real = wavefunction_pred_real.reshape(
                    (n_points, n_points)
                )
                wavefunction_pred_imag = wavefunction_pred_imag.reshape(
                    (n_points, n_points)
                )
                p_pred = wavefunction_pred_real**2  # + wavefunction_pred_imag**2

                time = time.reshape((n_points, n_points))
                wavefunction_gt_real = wavefunction_gt * np.cos(E_n * time / hbar)
                wavefunction_gt_imag = -wavefunction_gt * np.sin(E_n * time / hbar)
                p_gt = wavefunction_gt_real**2  # + wavefunction_gt_imag**2

                title = "{n},{l},{m} ORBITAL AT TIME t={timestep}".format(
                    n=n, l=l, m=m, timestep=time_point
                )
                self._plot(p_gt, p_pred, rmax, title)

        elif self.script == "TISE_hydrogen_without_decomposition":
            quantum_numbers = self.experiment_params["quantum_numbers"]
            n, l, m = quantum_numbers.values()
            R_nl_gt, f_lm_gt, g_m_gt = TISE_hydrogen_exact(r, theta, phi, n, l, m)
            wavefunction_gt = R_nl_gt * f_lm_gt * g_m_gt
            p_gt = wavefunction_gt**2  # Probability distribution

            r = r.reshape((n_points * n_points, 1))
            theta = theta.reshape((n_points * n_points, 1))
            phi = phi.reshape((n_points * n_points, 1))
            input_format = np.hstack((r, theta, phi))
            wavefunction_pred = model.predict(input_format)
            wavefunction_pred = wavefunction_pred[:, 0:1]  # real part
            # normalization
            normalization_constant = self.solver.normalize_output(model=model)
            wavefunction_pred = normalization_constant * wavefunction_pred
            ###############
            wavefunction_pred = wavefunction_pred.reshape((n_points, n_points))
            p_pred = wavefunction_pred**2

            title = "{n},{l},{m} ORBITAL".format(n=n, l=l, m=m)
            # self.solver.plot(p_gt, p_pred, rmax, title)
            self._plot(
                p_gt, p_pred, rmax, title
            )  # it makes sense to define plot function just once!

        elif self.script == "TISE_stark_effect":
            quantum_numbers = self.experiment_params["quantum_numbers"]
            electric_field = float(self.experiment_params["electric_field"])
            index = self.experiment_params["index"]
            n, m = quantum_numbers.values()

            eigenvalues, eigenvectors = TISE_stark_effect_exact(
                r, theta, phi, electric_field, n, m
            )
            p_gt = eigenvectors[index] ** 2  # Probability distribution

            r = r.reshape((n_points * n_points, 1))
            theta = theta.reshape((n_points * n_points, 1))
            phi = phi.reshape((n_points * n_points, 1))
            input_format = np.hstack((r, theta, phi))
            wavefunction_pred = model.predict(input_format)
            wavefunction_pred = wavefunction_pred[:, 0:1]  # real part
            # normalization
            normalization_constant = self.solver.normalize_output(model=model)
            wavefunction_pred = normalization_constant * wavefunction_pred
            ###############
            wavefunction_pred = wavefunction_pred.reshape((n_points, n_points))
            p_pred = wavefunction_pred**2

            ener = eigenvalues[index] / (-3 * e * electric_field * a0 / 2)
            title = (
                "{n},{m} MANIFOLD, ENERGY: {e}".format(n=n, m=m, e=np.rint(ener))
                + "$\Delta$E"
            )
            self._plot(p_gt, p_pred, rmax, title)

        elif self.script == "TISE_hydrogen":
            from TISE_hydrogen import (
                create_model_R_nl,
                create_model_f_lm,
                create_model_g_m,
            )

            R_params = self.network_params["R"]
            f_params = self.network_params["f"]
            g_params = self.network_params["g"]

            quantum_numbers = self.experiment_params["quantum_numbers"]
            n, l, m = quantum_numbers.values()
            R_nl_gt, f_lm_gt, g_m_gt = TISE_hydrogen_exact(r, theta, phi, n, l, m)
            wavefunction_gt = R_nl_gt * f_lm_gt * g_m_gt
            p_gt = wavefunction_gt**2  # Probability distribution

            r = r.reshape((n_points * n_points, 1))
            theta = theta.reshape((n_points * n_points, 1))
            phi = phi.reshape((n_points * n_points, 1))

            # R MODEL
            R_model, geom, data, net = create_model_R_nl(
                self.experiment_params, R_params
            )
            R_model.compile("adam", lr=1e-3, loss="MSE")
            with open(self.save_pth + "model_R/" + "best_step.txt") as f:
                best = f.readlines()[0]
            R_model.restore(
                self.save_pth + "model_R/" + "model.ckpt-" + best, verbose=1
            )
            R_nl_pred = R_model.predict(r)

            # normalization
            X = geom.uniform_points(1000)
            r_norm = X.reshape(-1)
            R_nl_norm = R_model.predict(X)
            R_nl_norm = R_nl_norm.reshape(-1)
            integrand = R_nl_norm**2 * r_norm**2
            C = simps(integrand, r_norm)
            normalization_constant_R = 1.0 / np.sqrt(C)
            R_nl_pred = R_nl_pred * normalization_constant_R
            tf.compat.v1.reset_default_graph()

            # f MODEL
            if f_params["prior"] == "Fourier":
                (
                    f_model,
                    geom,
                    data,
                    net,
                    prior_data,
                    prior_save_path,
                    compile_train_args,
                ) = create_model_f_lm(
                    self.experiment_params, f_params, self.results_path
                )
            else:
                f_model, geom, data, net = create_model_f_lm(
                    self.experiment_params, f_params, self.results_path
                )
            f_model.compile("adam", lr=1e-3, loss="MSE")
            with open(self.save_pth + "model_f/" + "best_step.txt") as f:
                best = f.readlines()[0]
            f_model.restore(
                self.save_pth + "model_f/" + "model.ckpt-" + best, verbose=1
            )
            f_lm_pred = f_model.predict(theta)

            # normalization
            X = geom.uniform_points(1000)
            theta_norm = X.reshape(-1)
            f_lm_norm = f_model.predict(X)
            f_lm_norm = f_lm_norm.reshape(-1)
            integrand = f_lm_norm**2 * np.sin(theta_norm)
            C = simps(integrand, theta_norm)
            normalization_constant_f = 1.0 / np.sqrt(C)
            f_lm_pred = f_lm_pred * normalization_constant_f
            tf.compat.v1.reset_default_graph()

            # g MODEL
            g_model, geom, data, net = create_model_g_m(
                self.experiment_params, g_params
            )
            g_model.compile("adam", lr=1e-3, loss="MSE")
            with open(self.save_pth + "model_g/" + "best_step.txt") as f:
                best = f.readlines()[0]
            g_model.restore(
                self.save_pth + "model_g/" + "model.ckpt-" + best, verbose=1
            )
            g_m_pred = g_model.predict(phi)

            # normalization
            X = geom.uniform_points(1000)
            phi_norm = X.reshape(-1)
            g_m_norm = g_model.predict(X)
            g_m_norm_real, g_m_norm_imag = g_m_norm[:, 0], g_m_norm[:, 1]
            integrand = g_m_norm_real**2 + g_m_norm_imag**2
            C = simps(integrand, phi_norm)
            normalization_constant_g = 1.0 / np.sqrt(C)
            g_m_pred = g_m_pred * normalization_constant_g
            tf.compat.v1.reset_default_graph()
            #####################

            wavefunction_pred = R_nl_pred * f_lm_pred * g_m_pred[:, 0:1]
            wavefunction_pred = wavefunction_pred.reshape((n_points, n_points))
            p_pred = wavefunction_pred**2

            title = "{n},{l},{m} ORBITAL".format(n=n, l=l, m=m)
            self._plot(p_gt, p_pred, rmax, title)

        elif self.script == "TDSE_one_level_AC":
            X_test = geom.uniform_points(10000)
            t = X_test[:, 0]
            F = self.experiment_params["F"]
            omega = self.experiment_params["omega"]
            d = self.experiment_params["d"]
            alpha = self.experiment_params["alpha"]

            # # for generalization evaluation
            # quasi_energy = - 0.25 * alpha * F**2
            # period = np.abs(2*pi / quasi_energy)
            # t += period
            # # t2 = t + period
            # # t = np.hstack([t, t2])
            # ###############################

            a_gt = TDSE_one_level_AC_exact(t, F, omega, d, alpha)
            t = t.reshape(-1, 1)
            a_pred = model.predict(t)

            self.solver.plot(t, a_gt, a_pred, save=False)

        tf.compat.v1.reset_default_graph()


def main():
    ### save_pth AND THE PATH GIVEN IN CONFIG FILE CAN BE DIFFERENT
    ### IF THE EXPERIMENTS ARE PERFORMED ON A CLUSTER
    # =========================
    # script = "TDSE_unperturbed"
    # save_pth = "model ckpt path"
    # config_path = "config file path"
    # =========================
    script = "TISE_hydrogen_without_decomposition"
    save_pth = "model ckpt path"
    config_path = "config file path"
    # =========================
    # script = "TISE_stark_effect"
    # save_pth = "model ckpt path"
    # config_path = "config file path"
    # =========================
    # script = "TISE_hydrogen"
    # save_pth = "model ckpt path"
    # config_path = "config file path"
    # =========================
    # script = "TDSE_one_level_AC"
    # save_pth = "model ckpt path"
    # config_path = "config file path"
    # =========================
    # script = "TDSE_laser_field"
    # save_pth = "model ckpt path"
    # config_path = "config file path"
    # =========================

    tester = PINNTest(script=script, save_pth=save_pth, config_path=config_path)
    tester.evaluate_metrics(sampled_points=10000)
    # tester.plots()
    # tester.evaluate_weights()


if __name__ == "__main__":
    main()
