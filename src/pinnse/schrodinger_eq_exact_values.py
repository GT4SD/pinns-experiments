import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, epsilon_0, hbar, pi, e, electron_mass, physical_constants
a0 = physical_constants['Bohr radius'][0]
from scipy.special import genlaguerre, lpmv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# TODO: change electron mass to reduced mass

# testing purposes
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

    g_m_real = np.cos(m*phi)
    g_m_imaginary = np.sin(m*phi)
    #g_m = np.vstack([g_m_real, g_m_imaginary])

    return R_nl, f_lm, g_m_real


def main():
    def func_R(x): # http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/hydwf.html#c3 --> in order to compare
                   # be careful with the prefactors of polar, asimuth
        factor = 4 / (81*np.sqrt(6)*a0**(3/2))
        return factor * (6 - (x/a0)) * (x/a0) * np.exp(-x/(3*a0))

    r = np.linspace(0, 25, 100)
    theta = np.linspace(0, pi, 100)
    phi = np.linspace(0, 2*pi, 100)

    R_nl, f_lm, g_m = TISE_hydrogen_exact(r, theta, phi, 3, 1, 0)


    ############################
    ## Radial function test ##
    ############################
    plt.figure(figsize=(12,8))
    plt.plot(r, func_R(r), 'ok')
    plt.plot(r, R_nl, '.r')
    plt.plot(r, r**2 * R_nl**2, '.b') # actually there also is a factor 4pi
    plt.show()
    ############################


    # ############################
    # ## Angular function test ##
    # ############################
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


    # ############################
    # ## Complete orbitals ##
    # ############################
    # # section of the orbitals with z=0 or y=0
    # rmax = 30
    # n_points = 1000 # must be an even number
    # x = np.linspace(-rmax,rmax,n_points)
    # # y = np.linspace(-rmax,rmax,n_points)
    # # X, Y = np.meshgrid(x, y)
    # # r = np.sqrt(X**2 + Y**2)
    # # theta = pi / 2   # since we are at z=0
    # # phi = np.arctan(Y / X)
    # z = np.linspace(-rmax,rmax,n_points)
    # X, Z = np.meshgrid(x, z)
    # r = np.sqrt(X**2 + Z**2)
    # theta = np.arctan(np.sqrt(X**2) / Z)
    # phi = [pi*np.ones([n_points,int(np.floor(n_points/2))])
    #     , np.zeros([n_points,int(np.ceil(n_points/2))])]
    # phi = np.hstack(phi)

    # R_nl, f_lm, g_m = TISE_hydrogen_exact(r, theta, phi, 3,1,0)
    # wavefunction = R_nl*f_lm*g_m
    # #p = r**2 * wavefunction**2
    # p = wavefunction**2

    # fig = plt.figure()
    # plt.imshow(p.T,extent=[-rmax, rmax, -rmax, rmax], interpolation='none',origin='lower', cmap=cm.inferno) 
    # plt.xlabel('x')
    # #plt.ylabel('y')
    # plt.ylabel('z')
    # plt.show()

    # TODO: careful with prefactors

    

# if __name__ == "__main__":
#     main()





