"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# MODEL OF REACTION DIFFUSION #
# https://rajeshrinet.github.io/blog/2016/gray-scott/

def pde(args, concentrations):
    # I cannot put params as an argument --> problem in pde.py
    # --> need to define auxiliary function
    params=[2*1e-5, 1e-5, 0.04, 0.1]
    '''
    https://www.chebfun.org/examples/pde/GrayScott.html
    args: (x, y, time)
    out: (u, v)
    params: list of pde parameters [ep1, ep2, b, d]
    '''
    u, v = concentrations[:, 0:1], concentrations[:, 1:2] # concentrations of reactants

    du_dt = dde.grad.jacobian(concentrations, args, i=0, j=2)
    dv_dt = dde.grad.jacobian(concentrations, args, i=1, j=2)

    du_dxx = dde.grad.hessian(concentrations, args, component=0, i=0, j=0)
    dv_dxx = dde.grad.hessian(concentrations, args, component=1, i=0, j=0)

    du_dyy = dde.grad.hessian(concentrations, args, component=0, i=1, j=1)
    dv_dyy = dde.grad.hessian(concentrations, args, component=1, i=1, j=1)

    ep1, ep2, b, d = params

    eq_1 = du_dt - ep1*(du_dxx+du_dyy) - b*(1-u) + u * v**2
    eq_2 = dv_dt - ep2*(dv_dxx+dv_dyy) + d*v - u * v**2

    return [eq_1, eq_2]

################################
def ic_u_func(args):
    x, y = args[:,0:1], args[:,1:2]
    return 1 - np.exp(-80*((x+0.05)**2 + (y+0.02)**2))
    
def ic_v_func(args):
    x, y = args[:,0:1], args[:,1:2]
    return np.exp(-80*((x-0.05)**2 + (y-0.02)**2))
################################

def main():
    geom = dde.geometry.Rectangle([-1,-1], [1,1])
    timedomain = dde.geometry.TimeDomain(0, 3500)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    ic_u = dde.IC(geomtime, ic_u_func, lambda _, on_initial: on_initial)
    ic_v = dde.IC(geomtime, ic_v_func, lambda _, on_initial: on_initial, component=1)

    data = dde.data.TimePDE(
        geomtime, pde, [ic_u, ic_v], num_domain=3500, num_boundary=400, num_initial=400)

    ## model ##
    #net = dde.maps.FNN([3] + [20] * 3 + [2], "tanh", "Glorot normal")
    net = dde.maps.FNN([3] + [100] * 3 + [50] * 2 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1.0e-3)
    model.train(epochs=10000)
    ############

    X = geomtime.random_points(1000000)
    conc_pred = model.predict(X)
    print(conc_pred.shape)
    u_concentration = conc_pred[:, 0]
    v_concentration = conc_pred[:, 1]

    results_path = '/Users/lat/Desktop/pinns/testing_deepXDE/deepxde/my_examples/results_gray_scott_2d/'
    np.savetxt(results_path+'X.txt', X)
    np.savetxt(results_path+'conc_pred.txt', conc_pred)
    # X = np.loadtxt(results_path+'X.txt', dtype=float)
    # conc_pred = np.loadtxt(results_path+'conc_pred.txt', dtype=float)

    ### Plotting ###

    indexes = np.argwhere((3500-X[:,2]) < 500).flatten()
    x_vals = X[:,0][indexes]
    y_vals = X[:,1][indexes]
    u_vals = u_concentration[indexes]

    plt.figure()
    plt.scatter(x_vals, y_vals, c=u_vals, cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('Concentration u after diffusion')

    plt.savefig(results_path + 'result.png')

def testing():
    results_path = '/Users/lat/Desktop/pinns/testing_deepXDE/deepxde/my_examples/results_gray_scott_2d/'
    X = np.loadtxt(results_path+'X.txt', dtype=float)
    conc_pred = np.loadtxt(results_path+'conc_pred.txt', dtype=float)

    timesteps = [0, 100,300, 500, 700,1000]

    # indexes = np.argwhere(np.abs(X[:,2] - 100) < 50).flatten()
    # x_vals = X[:,0][indexes]
    # y_vals = X[:,1][indexes]
    # v_concentration = conc_pred[:, 1]
    # v_vals = v_concentration[indexes]

    # plt.figure()
    # #plt.pcolor(x_vals, y_vals, v_vals, cmap=plt.cm.RdBu)
    # plt.scatter(x_vals, y_vals, c=v_vals, cmap='viridis')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.colorbar()
    # plt.title('Concentration u after diffusion')

    f = plt.figure()
    for i, t in enumerate(timesteps):
        indexes = np.argwhere(np.abs(X[:,2] - t) < 50).flatten()
        x_vals = X[:,0][indexes]
        y_vals = X[:,1][indexes]
        v_concentration = conc_pred[:, 1]
        v_vals = v_concentration[indexes]
        sp = f.add_subplot(3,2,i+1)
        plt.scatter(x_vals, y_vals, c=v_vals, cmap='viridis')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()


    plt.savefig(results_path + 'testing.png')


if __name__ == "__main__":
    #main()
    testing()

    
