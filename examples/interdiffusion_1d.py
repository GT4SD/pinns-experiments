import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepxde as dde
from scipy.special import erf

# FICK DIFFUSION EQUATION

c1, c2, D = 5, 1, 1e-2 #no idea how large these params must be :(((
# c1 and c2 initial concentrations on left and right side resp.
def pde(args, c):
    '''
    args[:,0]: spatial domain (x)
    args[:,1]: temporal domain (t)
    '''
    dc_dt = dde.grad.jacobian(c, args, j=1)
    dc_dxx = dde.grad.hessian(c, args, i=0, j=0)
    equation = dc_dt - D*dc_dxx

    return equation

# def ic_func(args):
#     '''
#     https://my.eng.utah.edu/~lzang/images/lecture-4.pdf --> case2: interdiffusion, constant c at x=0
#     '''
#     x = args[:,0:1]
#     return tf.where((x>0), c2, c1)

def ic_func(args):
    x = args[:,0:1]
    return c2 * tf.ones_like(x)

def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)
def bc_func(args):
    x = args[:,0:1]
    return c1 * tf.ones_like(x)

def solution_func(x, t):
    # average = (c1+c2)/2
    # return average - average * erf(x/(2.*np.sqrt(D*t)))
    return c1 - (c1-c2)*erf(x/(2.*np.sqrt(D*t)))

def main():
    results_path = '/Users/lat/Desktop/Code/pinns-experiments/examples/results_interdiffusion_1d/'
    geom = dde.geometry.Interval(0,1)
    timedomain = dde.geometry.TimeDomain(0,9)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    ic = dde.IC(geomtime, ic_func, lambda _, on_initial: on_initial)
    bc = dde.DirichletBC(geomtime, bc_func, boundary_l)

    # data can be created with the solution function as well!
    data = dde.data.TimePDE(geomtime, pde, [ic, bc], num_domain=700, num_boundary=200, num_initial=200)

    ## MODEL ##
    net = dde.maps.FNN([2] + [32] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1.0e-3)
    model.train(epochs=10000)
    ############

    X = geomtime.random_points(10000)
    # X = geomtime.uniform_points(10000)
    c_pred = model.predict(X)

    np.savetxt(results_path+'X.txt', X)
    np.savetxt(results_path+'c_pred.txt', c_pred)

    ## EVALUATION ##

    #time_points = [0, 0.25, 0.5, 0.75]
    time_points = [0, 3, 6, 9]

    f = plt.figure(figsize=(12,8))
    for i, t in enumerate(time_points):
        indexes = np.argwhere(np.isclose(X[:,1], t*np.ones(len(X[:,1])), atol=1e-1)).flatten()
        x_vals = X[:,0:1][indexes]
        t_vals = X[:,1:2][indexes]
        analytical = solution_func(x_vals, t_vals)
        predicted = c_pred[indexes]

        x_vals = x_vals.flatten()
        analytical = analytical.flatten()

        x_vals, predicted, analytical = zip(*sorted(zip(x_vals, predicted, analytical)))
        # instead of sorting, simply create data with: X = geomtime.uniform_points(10000)

        sp = f.add_subplot(2,2,i+1)
        plt.plot(x_vals, analytical, '-k', label='analytical')
        plt.plot(x_vals, predicted, '.r', label='predicted')
        plt.legend(loc='best')
        #plt.ylim(0, 5.1)
        plt.ylim(0.9, 5.1)
        plt.xlim(0, 1.1)
        plt.xlabel('x')
        plt.ylabel('c')
        plt.title('t = '+str(t))

    plt.savefig(results_path + 'results.png')


if __name__ == '__main__':
    main()


