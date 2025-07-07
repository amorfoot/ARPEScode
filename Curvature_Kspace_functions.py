from scipy import interpolate
import scipy.constants as const
from scipy.signal import savgol_filter
import numpy as np

from Helpful_functions import find_index

def crop_data(Xo, Yo, Zo, xlims, ylims):
    X_mask = (xlims[0]<Xo)&(Xo<xlims[1])
    Y_mask = (ylims[0]<Yo)&(Yo<ylims[1])
    return Xo[X_mask], Yo[Y_mask], Zo[Y_mask][:,X_mask]

def convert2kspace(angles, energies, data_2D, Ef, offset):

    new_angles = (angles - offset)*np.pi/180
    vec_y = np.sin(new_angles)
    # create the data set of sin(theta) by energy
    my_interp = interpolate.RectBivariateSpline(vec_y, energies, data_2D)

    E_min = np.min(energies)
    k_range = np.sqrt(E_min)*vec_y
    # for each energy strip evaluate only the angles which correspond to k_range
    new_data = np.zeros(data_2D.shape)
    for i, E in enumerate(energies):
        new_ang_vec = k_range/np.sqrt(E)
        new_data[:,i] = my_interp(new_ang_vec, E)[:,0]

    Energy = (energies-Ef)*1000 #center on Fermi energy and convert to meV
    f = np.sqrt(2*const.m_e*const.e)/const.hbar *1e-10 #convert to inverse angstrom
    k_range = k_range*f

    k_data_full = new_data.T
    k_full = k_range
    E_full = Energy

    return k_full, E_full, k_data_full

def reduce_k_data(k_full, E_full, k_data_full, dk=0.02, dE=3):
    
    '''Define the reduced X, Y, Z for the MDC set'''
    # Create X, Y, Z where the data now consists of MDC that have been integrated between dE
    E_step = np.mean(E_full[1:] - E_full[:-1])
    n_E = int(dE/E_step)
    # We want the reduced Y list to have a value as close as possible to 0meV
    target_i = find_index(0, E_full)
    indices = np.arange(E_full.size)
    i1 = E_full.size%n_E
    if i1 == 0:
        indices = indices
    else:
        indices = indices[:-i1]
    indices_red = np.mean([indices[i::n_E] for i in range(n_E)], axis=0)
    # find the largest index that is less than target_i
    close_i = max(indices_red[indices_red<target_i])
    dif = int(target_i - close_i)
    # dif defines the index such that when we reduce E_full[dif:] it will have a value close to 0meV

    E_full_ = E_full[dif:]
    k_data_full_ = k_data_full[dif:]

    i1 = E_full_.size%n_E
    if i1 == 0:
        Y = E_full_
        Z = k_data_full_
    else:
        Y = E_full_[:-i1]
        Z = k_data_full_[:-i1]

    X_M = k_full
    Y_M = np.mean([Y[i::n_E] for i in range(n_E)], axis=0)
    Z_M = np.mean([Z[i::n_E] for i in range(n_E)], axis=0)

    '''Define the reduced X, Y, Z for the MDC set'''
    # Create X, Y, Z where the data now consists of EDC that have been integrated between dk
    k_step = np.mean(k_full[1:] - k_full[:-1])
    n_k = int(dk/k_step)
    # We want the reduced Y list to have a value as close as possible to 0meV
    target_i = find_index(0, k_full)
    indices = np.arange(k_full.size)
    i1 = k_full.size%n_k
    if i1 == 0:
        indices = indices
    else:
        indices = indices[:-i1]
    indices_red = np.mean([indices[i::n_k] for i in range(n_k)], axis=0)
    # find the largest index that is less than target_i
    close_i = max(indices_red[indices_red<target_i])
    dif = int(target_i - close_i)
    # dif defines the index such that when we reduce E_full[dif:] it will have a value close to 0meV

    k_full_ = k_full[dif:]
    k_data_full_ = k_data_full[:,dif:]

    i1 = k_full_.size%n_k
    if i1 == 0:
        X = k_full_
        Z = k_data_full_
    else:
        X = k_full_[:-i1]
        Z = k_data_full_[:,:-i1]

    X_E = np.mean([X[i::n_k] for i in range(n_k)], axis=0)
    Y_E = E_full
    Z_E = np.mean([Z[:,i::n_k] for i in range(n_k)], axis=0)

    '''Define the reduced X, Y, Z for the 2D curvature set'''
    # Create X, Y, Z where the data now consists of EDC that have been integrated between dk
    # Although w use X_M, Y_M, Z_M
    k_step = np.mean(X_M[1:] - X_M[:-1])
    n_k = int(dk/k_step)
    # We want the reduced Y list to have a value as close as possible to 0meV
    target_i = find_index(0, X_M)
    indices = np.arange(X_M.size)
    i1 = X_M.size%n_k
    if i1 == 0:
        indices = indices
    else:
        indices = indices[:-i1]
    indices_red = np.mean([indices[i::n_k] for i in range(n_k)], axis=0)
    # find the largest index that is less than target_i
    close_i = max(indices_red[indices_red<target_i])
    dif = int(target_i - close_i)
    # dif defines the index such that when we reduce E_full[dif:] it will have a value close to 0meV

    X_M_ = X_M[dif:]
    Z_M_ = Z_M[:,dif:]

    i1 = X_M_.size%n_k
    if i1 == 0:
        X = X_M_
        Z = Z_M_
    else:
        X = X_M_[:-i1]
        Z = Z_M_[:,:-i1]

    X_ME = np.mean([X[i::n_k] for i in range(n_k)], axis=0)
    Y_ME = Y_M
    Z_ME = np.mean([Z[:,i::n_k] for i in range(n_k)], axis=0)

    return [X_M, Y_M, Z_M], [X_E, Y_E, Z_E], [X_ME, Y_ME, Z_ME]
      
def curvature_1D_M(X_M, Y_M, Z_M, f=5, dk=0.02):
    '''Curvature from the MDCs'''
    # Now we form the curvature and we define the Savitzky-Golay filter window as 5*dk
    xstep = np.mean(X_M[1:]-X_M[:-1])
    win_k = int(f*dk/xstep)

    # Build the arrays consisting of the 1st and 2nd derivatives of the MDCs
    M_2nd = np.zeros((Y_M.size, X_M.size))
    M_1st = np.zeros((Y_M.size, X_M.size))
    for i in range(Y_M.size):
        M_1st[i,:] = savgol_filter(Z_M[i], win_k, 3, deriv=1)
        M_2nd[i,:] = savgol_filter(Z_M[i], win_k, 3, deriv=2)
    M_1st = M_1st/dk
    M_2nd = M_2nd/(dk*dk)

    Mean, Max = np.mean(M_1st**2), np.max(M_1st**2)
    Co_k = Mean + (Max - Mean)
    X_C1D_M = X_M
    Y_C1D_M = Y_M
    Z_C1D_M = M_2nd/(Co_k + M_1st**2)**1.5
    
    return X_C1D_M, Y_C1D_M, Z_C1D_M, win_k

def curvature_1D_E(X_E, Y_E, Z_E, f=5, dE=3):
    '''Curvature from the EDCs'''
    # Now we form the curvature and we define the Savitzky-Golay filter window as 5*dE
    ystep = np.mean(Y_E[1:]-Y_E[:-1])
    win_E = int(f*dE/ystep)

    # Build the arrays consisting of the 1st and 2nd derivatives of the EDCs
    E_2nd = np.zeros((Y_E.size, X_E.size))
    E_1st = np.zeros((Y_E.size, X_E.size))
    for i in range(X_E.size):
        E_1st[:,i] = savgol_filter(Z_E[:,i], win_E, 3, deriv=1)
        E_2nd[:,i] = savgol_filter(Z_E[:,i], win_E, 3, deriv=2)
    E_1st = E_1st/dE
    E_2nd = E_2nd/(dE*dE)

    Mean, Max = np.mean(E_1st**2), np.max(E_1st**2)
    Co_E = Mean + (Max - Mean)
    X_C1D_E = X_E
    Y_C1D_E = Y_E
    Z_C1D_E = E_2nd/(Co_E + E_1st**2)**1.5
    
    return X_C1D_E, Y_C1D_E, Z_C1D_E, win_E

def curvature_2D(X, Y, Z, win=9):
    
    '''Curvature from the MDCs'''
    Xstep = np.mean(X[1:]-X[:-1])
    Ystep = np.mean(Y[1:]-Y[:-1])

    # Build the arrays consisting of the 1st and 2nd derivatives of the MDCs
    M_2nd = np.zeros((Y.size, X.size))
    M_1st = np.zeros((Y.size, X.size))
    for i in range(Y.size):
        M_1st[i,:] = savgol_filter(Z[i], win, 3, deriv=1)
        M_2nd[i,:] = savgol_filter(Z[i], win, 3, deriv=2)
    M_1st = M_1st/Xstep
    M_2nd = M_2nd/Xstep**2

    # Build the arrays consisting of the 1st and 2nd derivatives of the EDCs
    E_2nd = np.zeros((Y.size, X.size))
    E_1st = np.zeros((Y.size, X.size))
    for i in range(X.size):
        E_1st[:,i] = savgol_filter(Z[:,i], win, 3, deriv=1)
        E_2nd[:,i] = savgol_filter(Z[:,i], win, 3, deriv=2)
    E_1st = E_1st/Ystep
    E_2nd = E_2nd/Ystep**2
    
    EM_1st = np.zeros((Y.size, X.size))
    for i in range(X.size):
        EM_1st[:,i] = savgol_filter(M_1st[:,i], win, 3, deriv=1)
    EM_1st = EM_1st/Ystep

    Cx = 1/np.max(M_1st**2)
    Cy = Cx*(Ystep/Xstep)**2

    # 2D curvature
    Top_a = (1 + Cx*M_1st**2)*Cy*E_2nd
    Top_b = -2*Cy*Cx*M_1st*E_1st*EM_1st
    Top_c = (1 + Cy*E_1st**2)*Cx*M_2nd
    Bot = (1 + Cx*M_1st**2 + Cy*E_1st**2)**1.5
    C2D = (Top_a+Top_b+Top_c)/Bot
    
    return X, Y, C2D, win

'''Curvature
As defined in https://doi.org/10.1063/1.3585113'''