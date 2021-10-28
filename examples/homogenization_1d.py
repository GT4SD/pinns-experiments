import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde

# FICK DIFFUSION EQUATION
def pde(args, c):
    '''
    args[:,0]: spatial domain (x)
    args[:,1]: temporal domain (t)
    '''
    D = 1e-2
    dc_dt = dde.grad.jacobian(c, args, j=1)
    dc_dxx = dde.grad.hessian(c, args, i=0, j=0)
    equation = dc_dt - D*dc_dxx

    return equation


def ic_func(args):
    '''
    params = [c_mean, beta0, l]
    https://my.eng.utah.edu/~lzang/images/lecture-4.pdf --> case1: homogenization
    '''
    c_mean, beta0, l = 2, 1, 0.2 #no idea how large these params must be :(((
    x = args[:,0:1]
    return c_mean + beta0 * np.sin(np.pi * x / l)


def solution_func(x, t):
    c_mean, beta0, l = 2, 1, 0.2 #no idea how large these params must be :(((
    D = 1e-2
    tau = l**2 / (np.pi**2 * D)
    # x = args[:,0:1]
    # t = args[:,1:2]
    return c_mean + beta0 * np.sin(np.pi * x / l) * np.exp(-t / tau)


def main():
    #results_path = '/Users/lat/Desktop/pinns/testing_deepXDE/deepxde/my_examples/results_fick_diffusion_1d/'
    results_path = '/Users/lat/Desktop/Code/pinns-experiments/examples/results_homogenization_1d/'
    geom = dde.geometry.Interval(-1,1)
    timedomain = dde.geometry.TimeDomain(0,1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    ic = dde.IC(geomtime, ic_func, lambda _, on_initial: on_initial)

    # data can be created with the solution function as well!
    data = dde.data.TimePDE(geomtime, pde, [ic], num_domain=500, num_boundary=200, num_initial=200)

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

    time_points = [0, 0.25, 0.5, 0.75]

    f = plt.figure(figsize=(12,8))
    for i, t in enumerate(time_points):
        indexes = np.argwhere(np.isclose(X[:,1], t*np.ones(len(X[:,1])), atol=1e-2)).flatten()
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
        plt.xlabel('x')
        plt.ylabel('c')
        plt.title('t = '+str(t))
        plt.ylim(1,3)

    plt.savefig(results_path + 'results.png')


# if __name__ == '__main__':
#     main()
        




