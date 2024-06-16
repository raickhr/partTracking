import h5py
import numpy as np
import sys
import time
from itertools import cycle
from scipy.integrate import trapz
from scipy.signal import argrelmax, find_peaks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')
sys.stdout.flush()

# to construct a compact finite difference matrix to calculate derivatives
# the matrix size is (num, num)
# see S. K. Lele, Compact finite difference schemes with spectral-like resolution, 1992


def Create_matrix_fd(num):
    v1 = np.concatenate(([5.0, 2.0/11.0], np.ones(num-4)/3.0, [2.0/11.0]))
    B = np.diagflat(v1, 1) + np.diagflat(np.ones(num))
    v1 = np.concatenate(([2.0/11.0], np.ones(num-4)/3.0, [2.0/11.0, 5.0]))
    B = B + np.diagflat(v1, -1)
    A = np.zeros((num, num))
    A[0, :] = np.concatenate(
        ([-197.0/60, -5.0/12, 5.0, -5.0/3, 5.0/12, -1.0/20], np.zeros(num-6)))
    A[1, :] = np.concatenate(
        ([-20.0/33, -35.0/132, 34.0/33, -7.0/33, 2.0/33, -1.0/132], np.zeros(num-6)))
    A[num-2, :] = np.concatenate((np.zeros(num-6), [1.0 /
                                 132, -2.0/33, 7.0/33, -34.0/33, 35.0/132, 20.0/33]))
    A[num-1, :] = np.concatenate((np.zeros(num-6),
                                 [1.0/20, -5.0/12, 5.0/3, -5.0, 5.0/12, 197.0/60]))
    for ii in range(num-4):
        A[ii+2, :] = np.concatenate((np.zeros(ii), [-1.0 /
                                    36, -7.0/9, 0.0, 7.0/9, 1.0/36], np.zeros(num-ii-5)))
    return A, B

# get z derivative by compact finite difference


def calc_z_deri(rho, CFD_z):
    ny, nz = rho.shape
    result = np.zeros((ny, nz))
    for j in range(ny):
        result[j, :] = np.dot(CFD_z, rho[j, :])
    return result

# get x derivative by compact finite difference,
# note that in 2D RT flow I instead of x and z directions I am using y and z directions


def calc_x_deri(rho, CFD_x):
    ny, nz = rho.shape
    result = np.zeros((ny, nz))
    for j in range(nz):
        result[:, j] = np.dot(CFD_x, rho[:, j])
    return result

# plot the particle trajectory


def plot_traj(loc_all):
    SIZE = 12
    plt.rc('font', size=SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SIZE+4)  # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE+8)  # fontsize of the x any y labels
    plt.rc('xtick', labelsize=SIZE+2)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE+2)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE+5)   # legend fontsize

    fig = plt.figure(figsize=(6, 12))
    plt.xlim([0, 1.6])
    plt.ylim([0, 6.4])

    for j in range(loc_all.shape[1]):
        y = loc_all[:, j, 0]
        z = loc_all[:, j, 1]
        plt.plot(y, z)

# get Vy, Vz data from a particular grid location, n_vec is two integer numbers
# the mat parameter can be null, or can be '_16', '_32' to access the filtered velocity data


def query_point_velo(H5File, n_vec, step, mat=''):
    velo = np.zeros(2)
    velo[0] = H5File['/Fields/PVy' + mat + '/' +
                     str(step).zfill(6)][n_vec[1], n_vec[0], 0]
    velo[1] = H5File['/Fields/PVz' + mat + '/' +
                     str(step).zfill(6)][n_vec[1], n_vec[0], 0]
    return velo

# get data from HDF5 with group name called mat


def query_point_mat(H5File, mat, n_vec, step):
    return H5File['/Fields/P' + mat + '/' + str(step).zfill(6)][n_vec[1], n_vec[0], 0]

# interpolate velocity data from non-integer locations using Lagrangian 4th order polynomial
# see page 6 of http://turbulence.pha.jhu.edu/docs/Database-functions.pdf


def query_location_velo(H5File, loc, dl, Nx, Ny,  Nz, step, mat=''):
    loc_ny = int(loc[0]/dl[0] + 0.5)
    loc_nz = int(loc[1]/dl[1] + 0.5)

    # if...else... to account for particles outside domain
    # RT simulation y direction is periodic B.C., and z direction are walls
    if min(loc_ny, loc_nz) > 0 and loc_ny+2 < Ny and loc_nz+2 < Nz:
        pnts_veloy = H5File['/Fields/PVy' + mat + '/' + str(step).zfill(6)][loc_nz-1:loc_nz+3,
                                                                            loc_ny-1:loc_ny+3, 0]
        pnts_veloz = H5File['/Fields/PVz' + mat + '/' + str(step).zfill(6)][loc_nz-1:loc_nz+3,
                                                                            loc_ny-1:loc_ny+3, 0]
        pnts_veloy = np.squeeze(np.transpose(pnts_veloy))
        pnts_veloz = np.squeeze(np.transpose(pnts_veloz))
    else:
        pnts_veloy = np.zeros((4, 4))
        pnts_veloz = np.zeros((4, 4))

        for j in range(4):
            yloc = loc_ny - 1 + j
            if yloc < 0:
                yloc = Ny - 1
            elif yloc > Ny-1:
                yloc = yloc - Ny

            for k in range(4):
                zloc = loc_nz - 1 + k
                if zloc < 0:
                    zloc = 0
                elif zloc > Nz-1:
                    zloc = Nz-1

                n_vec = np.array([yloc, zloc])
                velo = query_point_velo(H5File, n_vec, step, mat)
                pnts_veloy[j, k] = velo[0]
                pnts_veloz[j, k] = velo[1]

    dest_vy = 0
    dest_vz = 0

    for j in range(4):
        for k in range(4):
            y_script = loc_ny - 1 + j
            z_script = loc_nz - 1 + k
            ly = 1.0
            lz = 1.0
            for ii in range(4):
                jj = loc_ny - 1 + ii
                if jj == y_script:
                    continue
                ly *= (loc[0]/dy - jj) / (y_script - jj)
            for ii in range(4):
                jj = loc_nz - 1 + ii
                if jj == z_script:
                    continue
                lz *= (loc[1]/dz - jj) / (z_script - jj)
            dest_vy += pnts_veloy[j, k] * ly * lz
            dest_vz += pnts_veloz[j, k] * ly * lz

    return np.array([dest_vy, dest_vz])


# interpolate the mat group
def query_location_mat(H5File, mat, loc, dl, Nx, Ny,  Nz, step):

    loc_ny = int(loc[0]/dl[0] + 0.5)
    loc_nz = int(loc[1]/dl[1] + 0.5)

    if min(loc_ny, loc_nz) > 0 and loc_ny+2 < Ny and loc_nz+2 < Nz:
        pnts_val = H5File['/Fields/P' + mat + '/' + str(step).zfill(6)][loc_nz-1:loc_nz+3,
                                                                        loc_ny-1:loc_ny+3, 0]
        pnts_mat = np.squeeze(np.transpose(pnts_val))
    else:
        pnts_mat = np.zeros((4, 4))

        for j in range(4):
            yloc = loc_ny - 1 + j
            if yloc < 0:
                yloc = Ny - 1
            elif yloc > Ny-1:
                yloc = yloc - Ny

            for k in range(4):
                zloc = loc_nz - 1 + k
                if zloc < 0:
                    zloc = 0
                elif zloc > Nz-1:
                    zloc = Nz-1

                n_vec = np.array([yloc, zloc])
                pnts_mat[j, k] = query_point_mat(H5File, mat, n_vec, step)

    dest_val = 0

    for j in range(4):
        for k in range(4):
            y_script = loc_ny - 1 + j
            z_script = loc_nz - 1 + k
            lx = 1.0
            ly = 1.0
            lz = 1.0
            for ii in range(4):
                jj = loc_ny - 1 + ii
                if jj == y_script:
                    continue
                ly *= (loc[0]/dy - jj) / (y_script - jj)
            for ii in range(4):
                jj = loc_nz - 1 + ii
                if jj == z_script:
                    continue
                lz *= (loc[1]/dz - jj) / (z_script - jj)

            dest_val += pnts_mat[j, k] * ly * lz

    return dest_val


# time_table is a 2d array where first row stores the step number in HDF5 file and 2nd row the corresponding time
# find the two neighboring steps and interpolate the information from them (LHS and RHS)
def get_velocity_time_table(H5File, loc, dl, Nx, Ny,  Nz, time_table, time, mat=''):
    time_position = np.searchsorted(time_table[1, :], time)
    if time_position == 0 or time_position == time_table.shape[1]:
        print('The time is out of range!')
        sys.exit()

    npoints = loc.shape[0]
    # LHS value
    time_l = time_table[1, time_position-1]
    velo_l = np.zeros(loc.shape)
    for i in range(npoints):
        velo_l[i, :] = query_location_velo(
            H5File, loc[i, :], dl, Nx, Ny, Nz, int(time_table[0, time_position-1]), mat)
    # RHS value
    time_r = time_table[1, time_position]
    velo_r = np.zeros(loc.shape)
    for i in range(npoints):
        velo_r[i, :] = query_location_velo(
            H5File, loc[i, :], dl, Nx, Ny, Nz, int(time_table[0, time_position]), mat)
    # LHS derivative
    if time_position == 1:
        dt_velo_l = (velo_r - velo_l) / \
            (time_table[1, time_position] - time_table[1, time_position-1])
    else:
        dt_velo_l = (velo_r - velo_l) / \
            (time_table[1, time_position] - time_table[1, time_position-1])
        velo_ll = np.zeros(loc.shape)
        for i in range(npoints):
            velo_ll[i, :] = query_location_velo(
                H5File, loc[i, :], dl, Nx, Ny, Nz, int(time_table[0, time_position-2]), mat)
        dt_velo_l += (velo_l - velo_ll) / \
            (time_table[1, time_position-1] - time_table[1, time_position-2])
        dt_velo_l *= 0.5

    # RHS derivative
    if time_position == time_table.shape[1]-1:
        dt_velo_r = (velo_r - velo_l) / \
            (time_table[1, time_position] - time_table[1, time_position-1])
    else:
        dt_velo_r = (velo_r - velo_l) / \
            (time_table[1, time_position] - time_table[1, time_position-1])
        velo_rr = np.zeros(loc.shape)
        for i in range(npoints):
            velo_rr[i, :] = query_location_velo(
                H5File, loc[i, :], dl, Nx, Ny, Nz, int(time_table[0, time_position+1]), mat)
        dt_velo_r += (velo_rr - velo_r) / \
            (time_table[1, time_position+1] - time_table[1, time_position])
        dt_velo_r *= 0.5

    #PCHIP: page 19 of http://turbulence.pha.jhu.edu/docs/Database-functions.pdf
    t_affine = (time - time_l) / (time_r - time_l)
    h00 = 2.0 * t_affine**3 - 3.0 * t_affine**2 + 1.0
    h10 = t_affine**3 - 2.0 * t_affine**2 + t_affine
    h01 = -2.0 * t_affine**3 + 3.0 * t_affine**2
    h11 = t_affine**3 - t_affine**2

    velo = h00 * velo_l + h10 * \
        (time_r - time_l) * dt_velo_l + h01 * \
        velo_r + h11 * (time_r - time_l) * dt_velo_r

    return velo


#interpolate mat group
def get_mat_time_table(H5File, mat, loc, dl, Nx, Ny,  Nz, time_table, time):
    time_position = np.searchsorted(time_table[1, :], time)
    if time_position == 0 or time_position == time_table.shape[1]:
        print('The time is out of range!')
        sys.exit()

    npoints = loc.shape[0]
    # LHS value
    time_l = time_table[1, time_position-1]
    mat_val_l = np.zeros(npoints)
    for i in range(npoints):
        mat_val_l[i] = query_location_mat(
            H5File, mat, loc[i, :], dl, Nx, Ny, Nz, int(time_table[0, time_position-1]))
    # RHS value
    time_r = time_table[1, time_position]
    mat_val_r = np.zeros(npoints)
    for i in range(npoints):
        mat_val_r[i] = query_location_mat(
            H5File, mat, loc[i, :], dl, Nx, Ny, Nz, int(time_table[0, time_position]))
    # LHS derivative
    if time_position == 1:
        dt_mat_val_l = (mat_val_r - mat_val_l) / \
            (time_table[1, time_position] - time_table[1, time_position-1])
    else:
        dt_mat_val_l = (mat_val_r - mat_val_l) / \
            (time_table[1, time_position] - time_table[1, time_position-1])
        mat_val_ll = np.zeros(npoints)
        for i in range(npoints):
            mat_val_ll[i] = query_location_mat(
                H5File, mat, loc[i, :], dl, Nx, Ny, Nz, int(time_table[0, time_position-2]))
        dt_mat_val_l += (mat_val_l - mat_val_ll) / \
            (time_table[1, time_position-1] - time_table[1, time_position-2])
        dt_mat_val_l *= 0.5

    # RHS derivative
    if time_position == time_table.shape[1]-1:
        dt_mat_val_r = (mat_val_r - mat_val_l) / \
            (time_table[1, time_position] - time_table[1, time_position-1])
    else:
        dt_mat_val_r = (mat_val_r - mat_val_l) / \
            (time_table[1, time_position] - time_table[1, time_position-1])
        mat_val_rr = np.zeros(npoints)
        for i in range(npoints):
            mat_val_rr[i] = query_location_mat(
                H5File, mat, loc[i, :], dl, Nx, Ny, Nz, int(time_table[0, time_position+1]))
        dt_mat_val_r += (mat_val_rr - mat_val_r) / \
            (time_table[1, time_position+1] - time_table[1, time_position])
        dt_mat_val_r *= 0.5

    t_affine = (time - time_l) / (time_r - time_l)
    h00 = 2.0 * t_affine**3 - 3.0 * t_affine**2 + 1.0
    h10 = t_affine**3 - 2.0 * t_affine**2 + t_affine
    h01 = -2.0 * t_affine**3 + 3.0 * t_affine**2
    h11 = t_affine**3 - t_affine**2

    mat_val = h00 * mat_val_l + h10 * \
        (time_r - time_l) * dt_mat_val_l + h01 * \
        mat_val_r + h11 * (time_r - time_l) * dt_mat_val_r

    return mat_val


# temporal integration of particle trajectory using predictor-corrector scheme
def update_position(H5File, loc, dl, Nx, Ny,  Nz, time_table, time_start, time_end, mat=''):
    # if time_start > time_end, backward tracking, else forward tracking
    dt = time_end - time_start
    vec_pred = get_velocity_time_table(
        H5File, loc, dl, Nx, Ny,  Nz, time_table, time_start, mat)
    loc_pred = loc + vec_pred * dt
    vec_corr = get_velocity_time_table(
        H5File, loc_pred, dl, Nx, Ny,  Nz, time_table, time_end, mat)
    loc_corr = loc + 0.5 * (loc_pred - loc) + 0.5 * dt * vec_corr
    return loc_corr

# get all trajectories, loc is the initial points, dl contains dx, dy, dz


def get_position(H5File, loc, dl, Nx, Ny,  Nz, time_table, time_start, time_end, dt, mat=''):
    # if time_start > time_end, backward tracking, else forward tracking
    num_incre = int(abs((time_end-time_start)/dt)) + 1
    dt = abs(dt) * np.where(time_end > time_start, 1, -1)
    time_series = time_start + np.arange(num_incre + 1) * dt
    time_series[num_incre] = time_end

    npoints = loc.shape[0]
    loc_curr = loc
    loc_all = np.zeros((time_series.shape[0], npoints, 2))
    loc_all[0, :, :] = loc
    time_curr = time_series[0]
    for i in range(time_series.shape[0]-1):
        time_next = time_series[i+1]
        loc_next = update_position(
            H5File, loc_curr, dl, Nx, Ny,  Nz, time_table, time_curr, time_next, mat)
        loc_all[i+1, :, :] = loc_next
        loc_curr = loc_next
        time_curr = time_next

    return loc_all, time_series

# get some specified information along trajectories, loc is the initial points, dl contains dx, dy, dz


def get_value_history(H5File, mat, loc, dl, Nx, Ny,  Nz, time_table, time_start, time_end, dt, new_mat=''):
    num_incre = int(abs((time_end-time_start)/dt)) + 1
    dt = abs(dt) * np.where(time_end > time_start, 1, -1)
    time_series = time_start + np.arange(num_incre + 1) * dt
    time_series[num_incre] = time_end

    npoints = loc.shape[0]
    loc_curr = loc
    loc_all = np.zeros((time_series.shape[0], npoints, 2))
    val_all = np.zeros((time_series.shape[0], npoints))
    loc_all[0, :, :] = loc
    val_all[0, :] = get_mat_time_table(
        H5File, mat, loc, dl, Nx, Ny,  Nz, time_table, time_start)
    time_curr = time_series[0]
    for i in range(time_series.shape[0]-1):
        time_next = time_series[i+1]
        loc_next = update_position(
            H5File, loc_curr, dl, Nx, Ny,  Nz, time_table, time_curr, time_next, new_mat)
        val_all[i+1, :] = get_mat_time_table(H5File, mat,
                                             loc_next, dl, Nx, Ny,  Nz, time_table, time_next)
        #print('At time ',time_next,', loc is ',loc_next)
        loc_all[i+1, :, :] = loc_next
        loc_curr = loc_next
        time_curr = time_next

    return val_all, loc_all, time_series

# similar to get_value_history, but stored as dictionary for convenience
# get some specified information along trajectories, loc is the initial points, dl contains dx, dy, dz


def get_value_history_dict(H5File, mat_list, loc, dl, Nx, Ny,  Nz, time_table, time_start, time_end, dt, new_mat=''):
    num_incre = int(abs((time_end-time_start)/dt)) + 1
    dt = abs(dt) * np.where(time_end > time_start, 1, -1)
    time_series = time_start + np.arange(num_incre + 1) * dt
    time_series[num_incre] = time_end

    npoints = loc.shape[0]
    loc_curr = loc
    loc_all = np.zeros((time_series.shape[0], npoints, 2))
    val_all_dict = {key: np.zeros(
        (time_series.shape[0], npoints)) for key in mat_list}
    loc_all[0, :, :] = loc
    for mat in mat_list:
        val_all_dict[mat][0, :] = get_mat_time_table(
            H5File, mat, loc, dl, Nx, Ny,  Nz, time_table, time_start)
    time_curr = time_series[0]
    for i in range(time_series.shape[0]-1):
        time_next = time_series[i+1]
        loc_next = update_position(
            H5File, loc_curr, dl, Nx, Ny,  Nz, time_table, time_curr, time_next, new_mat)
        for mat in mat_list:
            val_all_dict[mat][i+1, :] = get_mat_time_table(
                H5File, mat, loc_next, dl, Nx, Ny,  Nz, time_table, time_next)
        loc_all[i+1, :, :] = loc_next
        loc_curr = loc_next
        time_curr = time_next

    return val_all_dict, loc_all, time_series


input_file = '../2D_low/tests_multi_wo_wbar.h5'
input_file1 = '../2D_low_energy/tests_multi.h5'
time_file = 'time'
print('read the data')
sys.stdout.flush()

H5File = h5py.File(input_file, 'r')
H5File1 = h5py.File(input_file1, 'r')
Lx = 0.00625
Ly = 1.6
Lz = 6.4
Nx = 1
Ny = 256
Nz = 1024
dy = Ly/Ny
dz = Lz/Nz
dl = np.array([dy, dz])
time_table = np.loadtxt(time_file).T

print('get the initial points')
sys.stdout.flush()

selected_locs_rand_init = []
npoints = 500
for i in range(npoints):
    y_coord = np.random.uniform(0.8, 1.599)
    z_coord = np.random.uniform(2.5, 3.9)
    selected_locs_rand_init.append([y_coord, z_coord])
selected_locs_spiral_init = np.array(selected_locs_rand_init)

start_time = 0.1
end_time = 20
dt = 0.05

mat_list1 = ['rho_bar', 'rho_bar32', 'rho_bar64', 'Lambda_str', 'Lambda_str_32', 'Lambda_str_64',
             'Lambda_rot', 'Lambda_rot_32', 'Lambda_rot_64', 'KE_transport_16',
             'KE_transport_32', 'KE_transport_64', 'KE_transport_woP_16', 'KE_transport_woP_32', 'KE_transport_woP_64',
             'Press_contrib_16', 'Press_contrib_32', 'Press_contrib_64',
             'Press_dila_16', 'Press_dila_32', 'Press_dila_64',
             'Dissip_16', 'Dissip_32', 'Dissip_64', 'injection_16', 'injection_32', 'injection_64',
             'Vort_bar', 'Vort_bar_32', 'Vort_bar_64', 'W_tilde', 'W_tilde_32', 'W_tilde_64', 'Pi_trans_16',
             'Pi_trans_32', 'Pi_trans_64', 'Ens_dilate_16', 'Ens_dilate_32', 'Ens_dilate_64', 'Baro_omega_16',
             'Baro_omega_32', 'Baro_omega_64', 'Dissip_omega_16', 'Dissip_omega_32', 'Dissip_omega_64']
mat_name1 = ['rho_bar_val', 'rho_bar32_val', 'rho_bar64_val', 'lambda_str_val', 'lambda_str32_val', 'lambda_str64_val',
             'lambda_rot_val', 'lambda_rot32_val', 'lambda_rot64_val', 'KE_transport_val', 'KE_transport32_val',
             'KE_transport64_val', 'KE_transport_woP16_val', 'KE_transport_woP32_val', 'KE_transport_woP64_val',
             'Press_contrib16_val', 'Press_contrib32_val', 'Press_contrib64_val',
             'Press_dila_val', 'Press_dila32_val', 'Press_dila64_val', 'Dissip_val',
             'Dissip32_val', 'Dissip64_val', 'Inject_val', 'Inject32_val', 'Inject64_val', 'Vort_bar_val',
             'Vort_bar32_val', 'Vort_bar64_val', 'Vort_tilde_val', 'Vort_tilde32_val', 'Vort_tilde64_val',
             'Pi_trans_val', 'Pi_trans32_val', 'Pi_trans64_val', 'Ens_dilate_val', 'Ens_dilate32_val',
             'Ens_dilate64_val', 'Baro_omega_val', 'Baro_omega32_val', 'Baro_omega64_val', 'Dissip_omega_val',
             'Dissip_omega32_val', 'Dissip_omega64_val']


mat_list2 = ['Baro_approx_16', 'Baro_approx_32', 'Baro_approx_64', 'Press_stat_contrib_16',
             'Press_stat_contrib_32', 'Press_stat_contrib_64', 'Press_contrib_16', 'Press_contrib_32',
             'Press_contrib_64', 'KE_sub_16', 'KE_sub_32', 'KE_sub_64', 'KE_sub_trans_16', 'KE_sub_trans_32',
             'KE_sub_trans_64', 'Dissip_sub_16', 'Dissip_sub_32', 'Dissip_sub_64',
             'Lambda_16', 'Lambda_32', 'Lambda_64', 'Vy_tilde16', 'Vy_tilde32',
             'Vy_tilde64', 'Vz_tilde16', 'Vz_tilde32', 'Vz_tilde64', 'Pi_16', 'Pi_32', 'Pi_64']
mat_name2 = ['Baro_approx_contrib16_val', 'Baro_approx_contrib32_val', 'Baro_approx_contrib64_val',
             'Press_stat_contrib16_val', 'Press_stat_contrib32_val', 'Press_stat_contrib64_val',
             'Press_contrib16_val', 'Press_contrib32_val', 'Press_contrib64_val', 'KE_sub16_val', 'KE_sub32_val',
             'KE_sub64_val', 'KE_sub_trans16_val', 'KE_sub_trans32_val', 'KE_sub_trans64_val', 'Dissip_sub16_val',
             'Dissip_sub32_val', 'Dissip_sub64_val', 'lambda_val', 'lambda32_val', 'lambda64_val', 'Vy_tilde_val',
             'Vy_tilde32_val', 'Vy_tilde64_val', 'Vz_tilde_val', 'Vz_tilde32_val', 'Vz_tilde64_val', 'Pi_val',
             'Pi32_val', 'Pi64_val']

mat16_list1, mat32_list1, mat64_list1 = [], [], []

i = -1
for item in mat_list1:
    i = (i+1) % 3
    if i == 0:
        mat16_list1.append(item)
    elif i == 1:
        mat32_list1.append(item)
    elif i == 2:
        mat64_list1.append(item)

mat16_list2, mat32_list2, mat64_list2 = [], [], []

i = -1
for item in mat_list2:
    i = (i+1) % 3
    if i == 0:
        mat16_list2.append(item)
    elif i == 1:
        mat32_list2.append(item)
    elif i == 2:
        mat64_list2.append(item)


# velo_traj, _ = get_position(H5File, selected_locs_spiral_init, dl, Nx, Ny,  Nz, time_table, start_time, end_time, dt, '')
# velo_bar16_traj, _ = get_position(H5File, selected_locs_spiral_init, dl, Nx, Ny,  Nz, time_table, start_time, end_time, dt, '_tilde')
# velo_bar32_traj, _ = get_position(H5File, selected_locs_spiral_init, dl, Nx, Ny,  Nz, time_table, start_time, end_time, dt, '_tilde32')
# velo_bar64_traj, _ = get_position(H5File, selected_locs_spiral_init, dl, Nx, Ny,  Nz, time_table, start_time, end_time, dt, '_tilde64')

print('start to tracking data')
sys.stdout.flush()

all16_val1_dict, loc_all_random_16, time_series = get_value_history_dict(
    H5File, mat16_list1, selected_locs_spiral_init, dl, Nx, Ny,  Nz, time_table, start_time, end_time, dt, '_tilde')
print('finish data 1 tracking data 16')
sys.stdout.flush()
all32_val1_dict, loc_all_random_32, time_series = get_value_history_dict(
    H5File, mat32_list1, selected_locs_spiral_init, dl, Nx, Ny,  Nz, time_table, start_time, end_time, dt, '_tilde32')
print('finish data 1 tracking data 32')
sys.stdout.flush()
all64_val1_dict, loc_all_random_64, time_series = get_value_history_dict(
    H5File, mat64_list1, selected_locs_spiral_init, dl, Nx, Ny,  Nz, time_table, start_time, end_time, dt, '_tilde64')
print('finish data 1 tracking data 64')
sys.stdout.flush()


all16_val2_dict, loc_all_random_16, time_series = get_value_history_dict(
    H5File1, mat16_list2, selected_locs_spiral_init, dl, Nx, Ny,  Nz, time_table, start_time, end_time, dt, '_tilde')
print('finish data 2 tracking data 16')
sys.stdout.flush()
all32_val2_dict, loc_all_random_32, time_series = get_value_history_dict(
    H5File1, mat32_list2, selected_locs_spiral_init, dl, Nx, Ny,  Nz, time_table, start_time, end_time, dt, '_tilde32')
print('finish data 2 tracking data 32')
sys.stdout.flush()
all64_val2_dict, loc_all_random_64, time_series = get_value_history_dict(
    H5File1, mat64_list2, selected_locs_spiral_init, dl, Nx, Ny,  Nz, time_table, start_time, end_time, dt, '_tilde64')
print('finish data 2 tracking data 64')
sys.stdout.flush()

np.savez('lagrange_info_2d_ene_rand_pnts.npz', loc_all_random_16=loc_all_random_16, loc_all_random_32=loc_all_random_32,
         loc_all_random_64=loc_all_random_64, time_series=time_series,
         selected_locs_spiral_init=selected_locs_spiral_init, all16_val1_dict=all16_val1_dict,
         all32_val1_dict=all32_val1_dict, all64_val1_dict=all64_val1_dict, all16_val2_dict=all16_val2_dict,
         all32_val2_dict=all32_val2_dict, all64_val2_dict=all64_val2_dict)

print('finish tracking data')
sys.stdout.flush()
