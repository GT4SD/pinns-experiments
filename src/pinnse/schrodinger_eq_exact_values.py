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

# def TISE_hydrogen_exact(r, theta, phi, n=1, l=0, m=0, Z=1):
#     '''
#     https://www.bcp.fu-berlin.de/en/chemie/chemie/forschung/PhysTheoChem/agkeller/_Docs/TC_WS15_Ex07.pdf
#     give r, theta, phi as np.linspace
#     n, l, m are the quantum numbers
#     '''
#     R_prefactor = np.sqrt( ((2*Z)/(a0*n*hbar**2))**3 * np.math.factorial(n-l-1)/(2*n*np.math.factorial(n+l)) )
#     R_nl = R_prefactor * np.exp(-r*Z/(n*a0*hbar**2)) * (2*r*Z/(n*a0*hbar**2))**l * genlaguerre(n-l-1,2*l+1)(2*r*Z/(n*a0*hbar**2))
    
#     f_prefactor = (-1)**m * np.sqrt( (2*l+1)*np.math.factorial(l-m) / (4*pi*np.math.factorial(l+m)) )
#     f_lm = f_prefactor * lpmv(m,l,np.cos(theta))

#     g_m_real = np.cos(m*phi)
#     g_m_imaginary = np.sin(m*phi)
#     #g_m = np.vstack([g_m_real, g_m_imaginary])

#     return R_nl, f_lm, g_m_real

def TISE_hydrogen_exact(r, theta, phi, n=1, l=0, m=0, Z=1):
    '''
    https://www.bcp.fu-berlin.de/en/chemie/chemie/forschung/PhysTheoChem/agkeller/_Docs/TC_WS15_Ex07.pdf
    give r, theta, phi as np.linspace
    n, l, m are the quantum numbers
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

    
def main_1d(): # FOR 1D EQUATION
    x = np.linspace(-3, 3, 10000)
    # pl_1, mi_1 = TISE_hydrogen_1d_exact(x)
    # pl_2, mi_2 = TISE_hydrogen_1d_exact(x, 2)
    # pl_3, mi_3 = TISE_hydrogen_1d_exact(x, 3)
    # pl_4, mi_4 = TISE_hydrogen_1d_exact(x, 4)
    # results = [[pl_1, mi_1], [pl_2, mi_2], [pl_3, mi_3], [pl_4, mi_4]]

    f = plt.figure(figsize=(12,8))
    n_vals = [1,2,3,3]
    for i, n in enumerate(n_vals):

        sp = f.add_subplot(2,2,i+1)
        results_pl_re, results_pl_im = TISE_hydrogen_1d_momentum_space_exact(x, n=n)[0]
        zeros_re, zeros_im = TISE_hydrogen_1d_momentum_space_exact(0)[0]
        results_pl_re, results_pl_im = results_pl_re/zeros_re, results_pl_im/zeros_re
        plt.plot(x, results_pl_re, '.r', label='real part')
        plt.plot(x, results_pl_im, '.k', label='imaginary part')
        # plt.plot(pl_axis, results[i][0][500:], '.r')
        # plt.plot(mi_axis, results[i][1][:500], '.r')
        plt.legend(loc="best")
        plt.xlabel('p')
        plt.ylabel('phi')
        plt.title('n = '+str(n))

    plt.show()


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

    rmax = 30
    n_points = 100 # must be an even number
    x = np.linspace(-rmax,rmax,n_points)
    y = np.linspace(-rmax,rmax,n_points)
    z = np.linspace(-rmax,rmax,n_points)
    X, Y, Z = np.meshgrid(x, y, z)
    r = np.sqrt(X**2 + Z**2)
    theta = np.arctan(np.sqrt(X**2) / Z)
    phi = np.arctan(Y / X)

    R_nl, f_lm, g_m = TISE_hydrogen_exact(r, theta, phi, 4,2,0)
    wavefunction = R_nl*f_lm*g_m
    #p = r**2 * wavefunction**2
    p = wavefunction**2

    # checking the normalization!
    from scipy.integrate import simps
    integral = simps(simps(simps(p, x), y), z)
    print(integral)
    print(p.shape)



if __name__ == "__main__":
    main()
    #test()





