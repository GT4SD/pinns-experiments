import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass, physical_constants
a0 = physical_constants['Bohr radius'][0]
from scipy.special import genlaguerre, lpmv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# TODO: change electron mass to reduced mass

# testing purposes # TODO: we don't need to set everything to 0!, just be careful about the domain you're calculating!
hbar = 1
a0 = 1


def TISE_hydrogen_exact(r, theta, phi, n=1, l=0, m=0, Z=1):
    '''
    https://www.bcp.fu-berlin.de/en/chemie/chemie/forschung/PhysTheoChem/agkeller/_Docs/TC_WS15_Ex07.pdf
    - Give r, theta, phi as np.arrays, they must have the same shape. They represent the spherical values
        on the given grid (array).
    - n, l, m are the quantum numbers.
    - Only returns the real part of the orbitals.
    '''
    R_prefactor = np.sqrt( ((2*Z)/(a0*n))**3 * np.math.factorial(n-l-1)/(2*n*np.math.factorial(n+l)) )
    R_nl = R_prefactor * np.exp(-r*Z/(n*a0)) * (2*r*Z/(n*a0))**l * genlaguerre(n-l-1,2*l+1)(2*r*Z/(n*a0))
    
    f_prefactor = (-1)**m * np.sqrt( (2*l+1)*np.math.factorial(l-m) / (4*pi*np.math.factorial(l+m)) )
    f_lm = f_prefactor * lpmv(m,l,np.cos(theta))
    f_lm = f_lm * np.sqrt(2*pi) # for normalization

    g_m_real = np.cos(m*phi)
    g_m_imaginary = np.sin(m*phi)
    g_m_real = g_m_real / np.sqrt(2*pi) # for normalization
    #g_m = np.vstack([g_m_real, g_m_imaginary])

    return R_nl, f_lm, g_m_real


def TISE_stark_effect_exact(r, theta, phi, electric_field, n=2, m=0):
    '''
    http://www.physics.drexel.edu/~bob/Manuscripts/stark.pdf
    - n, m is the manifold to consider (So we have non-degenerate energy values!)
    - returns: eigenvalues, eigenvectors acted on r, theta, phi
    '''
    assert n>m, "n must have a greater value than m!"
    factor = - 3 * e * electric_field * a0 / 2
    m_abs = np.abs(m)
    if (n - m_abs) > 1:
        l_values = [x for x in range(m_abs+1, n)]
        M = np.zeros((len(l_values)+1, len(l_values)+1)) # initialize matrix
        for i,l in enumerate(l_values):
            R = n * np.sqrt((n-l) * (n+l))
            A = np.sqrt((l+m_abs) * (l-m_abs) / ((2*l +1) * (2*l -1)))
            M[i][i+1] = A * R
            M[i+1][i] = A * R
        w, v = np.linalg.eig(M) # Eigenvalues / Eigenvectors
        l_values = [x for x in range(m_abs, n)]
        basis_list = []
        for l in l_values:
            basis_list.append(TISE_hydrogen_exact(r, theta, phi, n=n, l=l, m=m))
        basis_list = [R*f*g for (R,f,g) in basis_list]
        basis_array = np.stack(basis_list, axis=-1)
        output_list = []
        for i in range(len(w)):
            val = np.matmul(basis_array, v[:,i])
            output_list.append(np.expand_dims(val, axis=0))
        eigenvectors = np.vstack(output_list)
        eigenvalues = factor * w
    else:
        eigenvalues = np.array([0])
        R, f, g = TISE_hydrogen_exact(r, theta, phi, n=n, l=m_abs, m=m)
        eigenvector = R*f*g
        eigenvectors = np.expand_dims(eigenvector, axis=0)
    return eigenvalues, eigenvectors



def TISE_hydrogen_1d_exact(x, n=1):
    '''
    http://theorie2.physik.uni-greifswald.de/member/bronold/SS2009/1DHatom.pdf
    functions 20a, 20b
    '''
    prefactor = np.sqrt(4 / (a0**2 * n**3))
    phi_n_plus = prefactor * (-1)**(n-1) * x * genlaguerre(n-1, 1)(2*x / (n*a0)) * np.exp(-x/(n*a0)) # defined for x>0
    phi_n_minus = prefactor * (-1)**(n) * x * genlaguerre(n-1, 1)(2*x / (n*a0)) * np.exp(x/(n*a0)) # defined for x<0
    return phi_n_plus, phi_n_minus


def TISE_hydrogen_1d_momentum_space_exact(p, n=1):
    '''
    http://theorie2.physik.uni-greifswald.de/member/bronold/SS2009/1DHatom.pdf
    functions 16a, 16b
    These functions are defined up to n=3
    '''
    assert n in [1,2,3], "Exact solution defined for only n=1,2,3"
    prefactor = np.sqrt(2*a0*n / (pi*hbar))
    same_factor = 1. / (1 + (n*p*a0/hbar)**2)
    alpha = n*p*a0 / hbar
    u, v = (1-alpha**2)/(1+alpha**2), (2*alpha)/(1+alpha**2)
    beta, gamma = (u**2 - v**2), 2*u*v
    if n==1:
        phi_n_plus_real = prefactor * same_factor * u
        phi_n_plus_imaginary = prefactor * same_factor * (-v)
        phi_n_minus_real = prefactor * same_factor * u
        phi_n_minus_imaginary = prefactor * same_factor * (v)
    if n==2:
        phi_n_plus_real = prefactor * same_factor * beta
        phi_n_plus_imaginary = prefactor * same_factor * (-gamma)
        phi_n_minus_real = prefactor * same_factor * beta
        phi_n_minus_imaginary = prefactor * same_factor * (gamma)
    if n==3: # TODO: implement this as recursive method for n>3
        beta, gamma = (beta**2 - gamma**2), 2*beta*gamma
        phi_n_plus_real = prefactor * same_factor * beta
        phi_n_plus_imaginary = prefactor * same_factor * (-gamma)
        phi_n_minus_real = prefactor * same_factor * beta
        phi_n_minus_imaginary = prefactor * same_factor * (gamma)
    return [[phi_n_plus_real, phi_n_plus_imaginary], [phi_n_minus_real, phi_n_minus_imaginary]]
    

def main():
    def func_R(x): # http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/hydwf.html#c3 --> in order to compare
                   # be careful with the prefactors of polar, asimuth
        factor = 4 / (81*np.sqrt(6)*a0**(3/2))
        return factor * (6 - (x/a0)) * (x/a0) * np.exp(-x/(3*a0))

    r = np.linspace(0, 25, 100)
    theta = np.linspace(0, pi, 100)
    phi = np.linspace(0, 2*pi, 100)

    R_nl, f_lm, g_m = TISE_hydrogen_exact(r, theta, phi, 2, 1, -1)


    ############################
    ## Radial function test ##
    ############################
    # from scipy.integrate import simps, quad
    # integral = simps(R_nl**2 * (r**2), r)
    # print(integral)
    # plt.figure(figsize=(12,8))
    # plt.plot(r, func_R(r), 'ok')
    # plt.plot(r, R_nl, '.r')
    # plt.plot(r, r**2 * R_nl**2, '.b') # actually there also is a factor 4pi
    # plt.show()
    ############################


    # ############################
    # ## Angular function test ##
    # ############################
    # from scipy.integrate import simps, quad
    # integral1 = simps(f_lm**2 * np.sin(theta), theta)
    # print(integral1)
    # integral2 = simps(g_m**2, phi)
    # print(integral2)
    # Y_lm = np.outer(g_m, f_lm) 
    # rho = np.abs(Y_lm)**2 # prob distribution of angular part
    # xs = 1 * np.outer(np.cos(phi), np.sin(theta)) #radius=1 
    # ys = 1 * np.outer(np.sin(phi), np.sin(theta))
    # zs = 1 * np.outer(np.ones(np.size(theta)), np.cos(theta))

    # color_map = cm.jet
    # scalarMap = cm.ScalarMappable(norm=plt.Normalize(vmin=np.min(rho),vmax=np.max(rho)), cmap=color_map)
    # C = scalarMap.to_rgba(rho) #scalarmap according to the #probability distribution stored in rho
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xs, ys, zs, rstride=2, cstride=2, color='b', facecolors=C) 
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    # ############################


    ############################
    ## Complete orbitals ##
    ############################
    # section of the orbitals with z=0 or y=0
    rmax = 30 
    n_points = 1000 # must be an even number
    x = np.linspace(-rmax,rmax,n_points)
    # y = np.linspace(-rmax,rmax,n_points)
    # X, Y = np.meshgrid(x, y)
    # r = np.sqrt(X**2 + Y**2)
    # theta = pi / 2   # since we are at z=0
    # phi = np.arctan(Y / X)
    z = np.linspace(-rmax,rmax,n_points)
    X, Z = np.meshgrid(x, z)
    r = np.sqrt(X**2 + Z**2)
    theta = np.arctan(np.sqrt(X**2) / Z)
    theta = np.where(theta<0,np.pi+theta,theta)
    phi = [pi*np.ones([n_points,int(np.floor(n_points/2))])
        , np.zeros([n_points,int(np.ceil(n_points/2))])]
    phi = np.hstack(phi)

    R_nl, f_lm, g_m = TISE_hydrogen_exact(r, theta, phi, 3,1,0)
    wavefunction = R_nl*f_lm*g_m
    #p = r**2 * wavefunction**2
    p = wavefunction**2
    print(np.max(R_nl))

    fig = plt.figure()
    plt.imshow(p,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot) 
    plt.xlabel('x')
    #plt.ylabel('y')
    plt.ylabel('z')
    plt.show()

    # fig, (ax1, ax2) = plt.subplots(1,2)
    # ax1.imshow(p.T,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot)
    # ax2.imshow(p.T,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot)
    # plt.show()

    # TODO: careful with prefactors


def test():
    # def integrand(x):
    #     out_r, out_i = TISE_hydrogen_1d_momentum_space_exact(x, n=1)[0]
    #     return out_r**2 + out_i**2
    # from scipy.integrate import quad
    # I = quad(integrand, -15, 15)
    # print(I)

    # r = np.linspace(0, 25, 100)
    # theta = np.linspace(0, pi, 100)
    # phi = np.linspace(0, 2*pi, 100)
    # R_nl, f_lm, g_m = TISE_hydrogen_exact(r, theta, phi, 3, 2, 0)
    # plt.figure(figsize=(12,8))
    # plt.plot(r, R_nl, '.r')
    # plt.show()

    rmax = 50
    n_points = 100 # must be an even number
    x = np.linspace(-rmax,rmax,n_points)
    y = np.linspace(-rmax,rmax,n_points)
    z = np.linspace(-rmax,rmax,n_points)
    X, Y, Z = np.meshgrid(x, y, z)
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arctan(np.sqrt(X**2 + Y**2) / Z)
    theta = np.where(theta<0,np.pi+theta,theta)
    phi = np.where(X<0, np.arctan(Y/X), np.arctan(Y/X)+pi)

    # R_nl, f_lm, g_m = TISE_hydrogen_exact(r, theta, phi, 4,2,0)
    # wavefunction = R_nl*f_lm*g_m
    # #p = r**2 * wavefunction**2
    # p = wavefunction**2

    # eigenvalues, eigenvectors = TISE_stark_effect_exact(r, theta, phi, electric_field=0, n=2, m=0)
    # wavefunction = eigenvectors[0]
    # p = wavefunction**2

    def solution_prior_20index0(r, theta, phi):
        alpha = 1 / (4*np.sqrt(2*pi))
        beta = alpha*r*np.exp(-r/(2*a0)) / (a0**(5/2))
        sol_w = beta * np.cos(theta) # 210
        sol_z = alpha * (1/(a0**(3/2))) * (2 - (r/a0)) * np.exp(-r/(2*a0)) # 200
        sol_u = (1/np.sqrt(2)) * (sol_w + sol_z)
        return sol_u
    wavefunction = solution_prior_20index0(r,theta,phi)
    p = wavefunction**2

    # checking the normalization!
    from scipy.integrate import simps
    integral = simps(simps(simps(p, x), y), z)
    print(integral)
    print(p.shape)


# def test():
#     limit = 30
#     n = 3
#     r = np.linspace(0,limit,10000)
#     R,f,g = TISE_hydrogen_exact(r,0,0,n,1,0)

#     from scipy.integrate import simps, quad
#     def func(x):
#         R,f,g = TISE_hydrogen_exact(x,0,0,n,1,0)
#         return R**2 * x**2
#     integrand = R**2 * r**2
#     C = simps(integrand, r)
#     print('simps: ', C)
#     C2 = quad(func, 0, limit)
#     print('quad: ', C2)
#     C3 = quad(func, 0, np.inf)
#     print('real: ', C3)


def main():
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

    electric_field = 10e6
    eigenvalues, eigenvectors = TISE_stark_effect_exact(r, theta, phi, electric_field, n=2, m=0) 
    # ----------------------

    print(eigenvalues / (- 3 * e * electric_field * a0 / 2) )
    from matplotlib import cm
    for i in range(len(eigenvalues)):
        # if i != 3:
        #     continue
        p = eigenvectors[i]**2
        fig = plt.figure()
        plt.imshow(p,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot) 
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('Probability distribution')
        plt.show()

    # # Test for n=4, m=0 TODO: passed
    # a = TISE_hydrogen_exact(r, theta, phi, n=4, l=0, m=0)
    # a = a[0] * a[1] * a[2]
    # #a = - np.sqrt(5/20) * a # -12
    # a = - np.sqrt(5/20) * a # 4
    # b = TISE_hydrogen_exact(r, theta, phi, n=4, l=1, m=0)
    # b = b[0] * b[1] * b[2]
    # #b = np.sqrt(9/20) * b
    # b = -np.sqrt(1/20) * b
    # c = TISE_hydrogen_exact(r, theta, phi, n=4, l=2, m=0)
    # c = c[0] * c[1] * c[2]
    # #c = - np.sqrt(5/20) * c
    # c = np.sqrt(5/20) * c
    # d = TISE_hydrogen_exact(r, theta, phi, n=4, l=3, m=0)
    # d = d[0] * d[1] * d[2]
    # #d = np.sqrt(1/20) * d
    # d = np.sqrt(9/20) * d
    # gt = a + b + c + d
    # p_gt = gt**2
    # fig = plt.figure()
    # plt.imshow(p_gt,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.hot) 
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.title('Probability distribution')
    # plt.show()

if __name__ == "__main__":
    #main()
    test()





