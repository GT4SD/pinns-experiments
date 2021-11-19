'''
PROBLEMS FROM: https://arxiv.org/pdf/2003.06496.pdf 
'''
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import tensorflow as tf

from abc import abstractmethod

from pinnse.custom_model import CustomLossModel
from pinnse.custom_boundary_conditions import PeriodicBC
from pinnse.custom_pde import CustomPDE
from pinnse.basis_functions import FourierBasis
from matplotlib import cm

def fourier_prior(args, f): # TODO: create a seperate script for priors
    x, y = args[:,0:1], args[:,1:2]
    u, v, p = f[:,0:1], f[:,1:2], f[:,2:3]
    basis = FourierBasis(max_k=10, dimension=2)
    representation = basis.compute(args)
    return [representation - u, representation - v, representation - p]


class SolverClass(object):

    def __init__(self, pinn_parameters, Re=40):
        self.net_architecture = pinn_parameters["net_architecture"]
        self.optimizer = pinn_parameters["optimizer"]
        self.lr = pinn_parameters["lr"]
        self.n_epochs = pinn_parameters["n_epochs"]
        self.Re = Re # Reynolds number
        assert len(self.lr) == len(self.n_epochs), "Length of lr and n_epochs must be same!"

    @abstractmethod
    def solve():
        """Solves the problem for the given PINN setting."""


class KovasznayFlow(SolverClass):

    def __init__(self, pinn_parameters):
        super().__init__(pinn_parameters)
        self.results_path = '/Users/lat/Desktop/Code/pinns-experiments/examples/results_navier_stokes/'

    def analytical_solution(self, x, y):
        x, y = np.meshgrid(x, y)
        nu = 1/self.Re
        Lambda = 1/(2.*nu) - np.sqrt(1/(4.*nu**2) + 4.*np.pi**2)

        u = 1 - np.exp(Lambda*x) * np.cos(2.*np.pi*y)
        v = (Lambda/(2.*np.pi)) * np.exp(Lambda*x) * np.sin(2.*np.pi*y)
        p = 1/2. * (1 - np.exp(2.*Lambda*x))
        return [u,v,p]

    def pde(self, args, out):
        """
        args = (x,y)
        out = (u, v, p) where u and v are the velocities in x and y direction resp., 
            p is the pressure of the system
        """
        nu = 1/self.Re
        u, v = out[:,0:1], out[:,1:2]

        du_dx = dde.grad.jacobian(out, args, i=0, j=0)
        du_dy = dde.grad.jacobian(out, args, i=0, j=1)
        du_dxx = dde.grad.hessian(out, args, component=0, i=0, j=0)
        du_dyy = dde.grad.hessian(out, args, component=0, i=1, j=1)

        dv_dx = dde.grad.jacobian(out, args, i=1, j=0)
        dv_dy = dde.grad.jacobian(out, args, i=1, j=1)
        dv_dxx = dde.grad.hessian(out, args, component=1, i=0, j=0)
        dv_dyy = dde.grad.hessian(out, args, component=1, i=1, j=1)

        dp_dx = dde.grad.jacobian(out, args, i=2, j=0)
        dp_dy = dde.grad.jacobian(out, args, i=2, j=1)

        eq1 = u*du_dx + v*du_dy + dp_dx - nu*(du_dxx + du_dyy)
        eq2 = u*dv_dx + v*dv_dy + dp_dy - nu*(dv_dxx + dv_dyy)
        eq3 = du_dx + dv_dy

        return [eq1, eq2, eq3]

    def boundary(self, x, on_boundary):
        return on_boundary

    def solve(self):
        self.geom = dde.geometry.Rectangle((-0.5, -0.5), (1.0, 1.5))
        bc_u = dde.NeumannBC(self.geom, lambda x: 0, self.boundary, component=0)
        bc_v = dde.NeumannBC(self.geom, lambda x: 0, self.boundary, component=1)

        self.data = dde.data.PDE(self.geom, self.pde, [bc_u, bc_v], num_domain=2600, num_boundary=400)
        net = dde.maps.FNN(self.net_architecture, "tanh", "Glorot uniform")

        # ----------- BASIC SETUP ------------
        self.model = dde.Model(self.data, net)
        # ----------- PRIOR LEARNING ------------
        # prior_data = dde.data.PDE(self.geom, fourier_prior, bcs=[], num_domain=500, num_boundary=200)
        # prior_save_path = self.results_path + "prior/model"
        # compile_train_args = {'optimizer':'adam', 'lr':1e-3, 'epochs':10000}
        # self.model = CustomLossModel(self.data, net)
        # self.model.learn_prior(prior_data, prior_save_path, **compile_train_args)
        # ---------------------------------
        for i in range(len(self.lr)):
            # ----------- BASIC SETUP ------------
            self.model.compile(self.optimizer, lr=self.lr[i])
            # ----------- PRIOR LEARNING ------------
            # self.model.compile(self.optimizer, lr=self.lr[i], loss='MSE')
            # ---------------------------------

            self.model.train(epochs=self.n_epochs[i])

    def relative_L2_error(self, gt, predicted):
        return np.sqrt(np.sum((predicted - gt)**2)) / np.sqrt(np.sum(gt**2))

    def evaluate(self):
        X = self.geom.uniform_points(10000)
        x, y = np.unique(X[:,0:1]), np.unique(X[:,1:2])
        gt = self.analytical_solution(x, y)
        predicted = self.model.predict(X)
        np.savetxt(self.results_path+'X.txt', X)
        np.savetxt(self.results_path+'predicitions.txt', predicted)
        u_pr, v_pr, p_pr = predicted[:,0:1], predicted[:,1:2], predicted[:,2:3]
        u_pr, v_pr, p_pr = u_pr.reshape(len(x), len(y)).T, v_pr.reshape(len(x), len(y)).T, p_pr.reshape(len(x), len(y)).T

        ## PRINTING ERRORS ##
        print("--- L2 ERRORS ---")
        print("u error: ", self.relative_L2_error(gt[0], u_pr))
        print("v error: ", self.relative_L2_error(gt[1], v_pr))
        print("p error: ", self.relative_L2_error(gt[2], p_pr))

        fig, (ax1, ax2, ax3) = plt.subplots(3,2)
        ax1[0].imshow(gt[0],extent=[np.min(x), np.max(x), np.min(y), np.max(y)], interpolation='none',origin='lower', cmap=cm.hot)
        ax1[0].title.set_text('Exact solution of u')
        ax1[0].set_xlabel('x')
        ax1[0].set_ylabel('y')
        ax1[1].imshow(u_pr,extent=[np.min(x), np.max(x), np.min(y), np.max(y)], interpolation='none',origin='lower', cmap=cm.hot)
        ax1[1].title.set_text('PINN prediction of u')
        ax1[1].set_xlabel('x')
        ax1[1].set_ylabel('y')

        ax2[0].imshow(gt[1],extent=[np.min(x), np.max(x), np.min(y), np.max(y)], interpolation='none',origin='lower', cmap=cm.hot)
        ax2[0].title.set_text('Exact solution of v')
        ax2[0].set_xlabel('x')
        ax2[0].set_ylabel('y')
        ax2[1].imshow(v_pr,extent=[np.min(x), np.max(x), np.min(y), np.max(y)], interpolation='none',origin='lower', cmap=cm.hot)
        ax2[1].title.set_text('PINN prediction of v')
        ax2[1].set_xlabel('x')
        ax2[1].set_ylabel('y')

        ax3[0].imshow(gt[2],extent=[np.min(x), np.max(x), np.min(y), np.max(y)], interpolation='none',origin='lower', cmap=cm.hot)
        ax3[0].title.set_text('Exact solution of p')
        ax3[0].set_xlabel('x')
        ax3[0].set_ylabel('y')
        ax3[1].imshow(p_pr,extent=[np.min(x), np.max(x), np.min(y), np.max(y)], interpolation='none',origin='lower', cmap=cm.hot)
        ax3[1].title.set_text('PINN prediction of p')
        ax3[1].set_xlabel('x')
        ax3[1].set_ylabel('y')

        plt.savefig(self.results_path + 'KovasznayFlow.png')

# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------

def main():
    pinn_parameters = {"net_architecture":[2] + [32] * 3 + [3], "optimizer":"adam", "lr":[1e-3, 1e-4, 1e-5, 1e-6], "n_epochs":[5000, 5000, 50000, 50000]}
    solver = KovasznayFlow(pinn_parameters=pinn_parameters)
    solver.solve()
    solver.evaluate()

if __name__ == "__main__":
    main()
