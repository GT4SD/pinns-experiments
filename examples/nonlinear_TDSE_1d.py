"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def pde(args, h):
    '''
    args: input arguments [x, t] --> spatial, temporal dimensions
    h: solution function h = [u,v] where u and v are the real and imaginary part resp.
    '''
    u, v = h[:, 0:1], h[:, 1:2]

    du_dt = dde.grad.jacobian(h, args, i=0, j=1)
    dv_dt = dde.grad.jacobian(h, args, i=1, j=1)

    du_dxx = dde.grad.hessian(h, args, component=0, i=0, j=0)
    dv_dxx = dde.grad.hessian(h, args, component=1, i=0, j=0)

    real_part = (-1)*dv_dt + 0.5*du_dxx + u**3 + u*(v**2)
    imaginary_part = du_dt + 0.5*dv_dxx + v**3 + v*(u**2)

    return [real_part, imaginary_part]




geom = dde.geometry.Interval(-5, 5)
timedomain = dde.geometry.TimeDomain(0, np.pi / 2)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], -5)

bc_0 = dde.PeriodicBC(geomtime, 0, boundary_left)
bc_1 = dde.PeriodicBC(geomtime, 0, boundary_left, derivative_order=1)
bc_0_2 = dde.PeriodicBC(geomtime, 0, boundary_left, component=1)
bc_1_2 = dde.PeriodicBC(geomtime, 0, boundary_left, derivative_order=1, component=1)
# 0 indicating args[0] coordinate --> spatial domain

def func(x):
    return 2.0 * (1 / np.cosh(x[:, 0:1]))

ic = dde.IC(geomtime, func, lambda _, on_initial: on_initial)
ic_2 = dde.IC(geomtime, func, lambda _, on_initial: on_initial, component=1)
# for IC, the component denotes the spatial, NOT the temporal argument!


############################################
##### DATA GENERATION / MODEL TRAINING #####
############################################

bc_list = [bc_0, bc_1, bc_0_2, bc_1_2, ic, ic_2]
data = dde.data.TimePDE(
    geomtime, pde, bc_list, num_domain=2500, num_boundary=200, num_initial=200
)

net = dde.maps.FNN([2] + [20] * 3 + [2], "tanh", "Glorot normal")
# fully-connected NN, may change the architecture, Glorot normal is the kernel initializer
# 2-dim output for the real / imaginary parts of the function
model = dde.Model(data, net)

model.compile("adam", lr=1.0e-3)
model.train(epochs=10000)
#model.compile("L-BFGS")
#model.train()

# copied from Burgers_RAR, adam and L-BFGS are optimizers
# TODO: learn about L-BFGS

# THERE WAS ADDITIONAL TRAINING STEPS, WHICH WE OMIT FOR THE TIME BEING:
# X = geomtime.random_points(100000)
# err = 1
# while err > 0.005:
#     f = model.predict(X, operator=pde)
#     err_eq = np.absolute(f)
#     err = np.mean(err_eq)
#     print("Mean residual: %.3e" % (err))

#     x_id = np.argmax(err_eq)
#     print("Adding new point:", X[x_id], "\n")
#     data.add_anchors(X[x_id])
#     early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
#     model.compile("adam", lr=1e-3)
#     model.train(epochs=10000, disregard_previous_best=True, callbacks=[early_stopping])
#     model.compile("L-BFGS")
#     losshistory, train_state = model.train()
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X = geomtime.random_points(10000)
# print(X.shape) # (10000, 2) first dim is spatial second is temporal
# print(max(X[:,0]), min(X[:,0]))
# print(max(X[:,1]), min(X[:,1]))
y_pred = model.predict(X)
print(y_pred.shape)
y_pred_real = y_pred[:, 0]
y_pred_imaginary = y_pred[:, 1]
h_absolute = np.sqrt(y_pred_real**2 + y_pred_imaginary**2)

results_path = '/Users/lat/Desktop/pinns/testing_deepXDE/deepxde/my_examples/results_nonlinear_TDSE_1d/'
np.savetxt(results_path+'X.txt', X)
np.savetxt(results_path+'y_pred.txt', y_pred)
# X = np.loadtxt(results_path+'X.txt', dtype=float)
# y_pred = np.loadtxt(results_path+'y_pred.txt', dtype=float)


##################################################
##### PLOTTING THE PREDICTED SOLUTION (REAL) #####
##################################################

# 3D PLOT --> looks like shit since it's not interactive
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# ax.scatter3D(X[:,1], X[:,0], h_absolute)

# ax.set_xlabel('t')
# ax.set_ylabel('x')
# ax.set_zlabel('|h|')

# plt.savefig('result.png')

tolerance = 0.02
i0 = np.argwhere((X[:,1]-0.59) < tolerance).flatten()
i1 = np.argwhere((X[:,1]-0.79) < tolerance).flatten()
i2 = np.argwhere((X[:,1]-0.98) < tolerance).flatten()

names = ['0.59','0.79','0.98']

counter = 0
for i in [i0, i1, i2]:
    x_vals = X[:,0][i]
    h_abs_vals = h_absolute[i]
    
    fig = plt.figure()
    plt.plot(x_vals, h_abs_vals, '.r')
    plt.xlabel('x')
    plt.ylabel('|h(t,x)|')
    plt.title('t = {name}'.format(name = names[counter]))

    plt.savefig(results_path + 'result_{name}.png'.format(name = names[counter]))
    counter += 1





