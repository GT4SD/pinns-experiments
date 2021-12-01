import math
import numpy as np
import tensorflow as tf


# TODO: add various test functinos --> allow unique test function definitions
# TODO: integrate only possible on 2d spatiotemporal rectangular grid --> extend it to arbirary n-dim domains!

# TODO: remove this commented code after testing
# def sin(x,k):
#     return np.sin(np.pi*x*k)

# def integrate_2d_spatiotemporal_grid(X, y, K, integration_method='trapezoid', test_function='sine'):
#     '''
#     only supported types:
#     integration_method = 'trapezoid'
#     test_function = 'sine' or 'unity'
#     '''
#     axis1, axis2 = np.unique(X[:,0]), np.unique(X[:,1])
#     dx1 = axis1[-1]-axis1[-2]
#     dx2 = axis2[-1]-axis2[-2]
#     area = dx1*dx2

#     y = np.transpose(y.reshape(shape2, shape1))

#     if test_function == 'sine':
#         func = sin

#     if integration_method == 'trapezoid':
#         matrix = 4.*np.ones((shape1, shape2))
#         for i in range(shape1):
#             for j in range(shape2):
#                 if (i==0 and j==0) or (i==0 and j==shape2-1) or (i==shape1-1 and j==shape2-1) or (i==shape1-1 and j==0):
#                     matrix[i][j] = 1
#                 elif (i==0) or (j==0) or (j==shape2-1) or (i==shape1-1):
#                     matrix[i][j] = 2

#     integral = 0
#     for k1 in range(1, K+1):
#         for k2 in range(1, K+1):
#             val1 = func(axis1, k1)
#             val2 = func(axis2, k2)
#             values = np.dot(val1.reshape([-1,1]), val2.reshape([1,-1]))
#             integrand = y * values
#             integrand = np.sum( integrand * matrix )
#             integral += integrand * area/4
#             if test_function == 'unity':
#                 return integral

#     return integral / K

def sin(x,k):
    pi = tf.constant(math.pi, dtype=x.dtype)
    return tf.sin(pi*x*k)
def unity(x,k):
    length = tf.shape(x)[0]
    return tf.ones(length, dtype=x.dtype)

def integrate_2d_spatiotemporal_grid(X, y, K, integration_method='trapezoid', test_function='sine'):
    '''
    only supported types:
    integration_method = 'trapezoid'
    test_function = 'sine' or 'unity'
    '''
    axis1 , _ = tf.unique(X[:,0])
    axis2 , _ = tf.unique(X[:,1])
    dx1 = axis1[-1]-axis1[-2]
    dx2 = axis2[-1]-axis2[-2]
    area = dx1*dx2

    # shape1, shape2 = axis1.shape[0], axis2.shape[0]
    shape1, shape2 = tf.shape(axis1)[0], tf.shape(axis1)[0]

    y_reshaped = tf.transpose(tf.reshape(y, (shape2, shape1)))
    y_reshaped = tf.transpose(tf.reshape(y, (18, 36)))
    #print('y_reshaped type : ', type(y_reshaped))

    if test_function == 'sine':
        func = sin
    elif test_function == 'unity':
        func = unity

    # TODO: uncomment
    # if integration_method == 'trapezoid':
    #     # matrix = 4.*tf.ones((shape1, shape2))
    #     # matrix_tensor = tf.Variable(matrix)
    #     # for i in range(range1):
    #     #     for j in range(range2):
    #     #         if (i==0 and j==0) or (i==0 and j==range2-1) or (i==range1-1 and j==range2-1) or (i==range1-1 and j==0):
    #     #             matrix[i][j] = 1
    #     #         elif (i==0) or (j==0) or (j==range2-1) or (i==range1-1):
    #     #             matrix[i][j] = 2
    #     # matrix_tensor.assign(matrix)
    #     matrix_tensor = 4.*tf.ones((shape1, shape2))
    #     matrix_tensor = tf.Variable(matrix_tensor)
    #     matrix_tensor[:,0].assign(2*tf.ones(shape1, dtype=y.dtype))
    #     matrix_tensor[:,-1].assign(2*tf.ones(shape1, dtype=y.dtype))
    #     matrix_tensor[0,:].assign(2*tf.ones(shape2, dtype=y.dtype))
    #     matrix_tensor[-1,:].assign(2*tf.ones(shape2, dtype=y.dtype))
    #     matrix_tensor[0,0].assign(1)
    #     matrix_tensor[-1,0].assign(1)
    #     matrix_tensor[0,-1].assign(1)
    #     matrix_tensor[-1,-1].assign(1)

    #integral = tf.Variable(0, dtype=y.dtype)
    integral = []
    for k1 in range(1, K+1):
        for k2 in range(1, K+1):
            val1 = func(axis1, k1)
            val2 = func(axis2, k2)
            val1 = tf.reshape(val1, (-1,1))
            val2 = tf.reshape(val2, (1,-1))
            values = tf.matmul(val1, val2)
            # val2 = tf.reshape(val2, (-1,1))
            # values = tf.matmul(val1, tf.transpose(val2))
            integrand = y_reshaped * values
            #print('integrand type before r_s : ', type(integrand))
            # TODO: integrand = tf.reduce_sum( integrand * matrix_tensor )
            #integral = integral.assign_add(integrand * area/4)
            # print('matrix_tensor type : ', type(matrix_tensor))
            # print('values type : ', type(values))
            # print('integrand type : ', type(integrand))
            # print('integral type : ', type(integrand))
            integral.append(integrand * area/4)
    integral = tf.stack(integral)

    return integral

# def integrate_2d_spatiotemporal_grid(X, y, K, integration_method='trapezoid', test_function='sine'):
#     return tf.zeros(7)


def integrate_1d(X, y):
    dx1 = tf.concat([X[1:], tf.reshape(X[-1], (1,1))], axis=0) - X
    dx2 = X - tf.concat([tf.reshape(X[0], (1,1)), X[:-1]], axis=0)

    integral = (y*dx1 + y*dx2)/2
    integral = tf.math.reduce_sum(integral)

    return integral



def main():
    '''
    Defining main for testing purposes
    '''
    def func(X):
        return tf.math.cos(X) * tf.math.exp(3.*X)
    X = tf.reshape(tf.Variable(np.linspace(-5,5,10000)), (-1,1))
    y = func(X)
    print(integrate_1d(X, y))


# if __name__ == "__main__":
#     main()