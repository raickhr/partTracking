import numpy as np

def gethorizontal_grad(pm, pn, phi):
    taxis = 0
    zaxis = 1
    yaxis = 2
    xaxis = 3
    phi_xplus =  0.5 * (np.roll(phi, -1, axis=xaxis) + phi)
    phi_xminus =  0.5 * (np.roll(phi, 1, axis=xaxis) + phi)
    phi_yplus =  0.5 * (np.roll(phi, -1, axis=yaxis) + phi)
    phi_yminus =  0.5 * (np.roll(phi, 1, axis=yaxis) + phi)

    grad_x = pm * (phi_xplus - phi_xminus)
    grad_y = pm * (phi_yplus - phi_yminus)
    return grad_x, grad_y


def get_vertical_gradient(phi, s_w):
    t_axis, z_axis, y_axis, x_axis = 0,1,2,3
    grad = np.zeros(phi.shape, dtype=np.float64)
    dz = np.roll(s_w,-1) - s_w
    dz = dz[0:-1]
    phi_plus = 0.5 * (np.roll(phi,-1, axis=z_axis) + phi)
    phi_minus = 0.5 * (np.roll(phi,1, axis=z_axis) + phi)
    dphi = phi_plus - phi_minus
    dphi[:,0,:,:] = dphi[:,1,:,:]
    dphi[:,-1,:,:] = dphi[:,-2,:,:]
    for i in range(len(dz)):
        grad[:,i,:,:] = dphi[:,i,:,:]/dz[i]
    return grad

def get_vertical_gradient_w(phi_w, s_w):
    t_axis, z_axis, y_axis, x_axis = 0,1,2,3
    tlen, zlen, ylen, xlen = phi_w.shape
    grad = np.zeros((tlen, zlen-1, ylen, xlen), dtype=np.float64)
    dz = np.roll(s_w,-1) - s_w
    dz = dz[0:-1]
    phi_w_minus = np.roll(phi_w,1, axis=z_axis)
    dphi = phi_w - phi_w_minus
    dphi = dphi[:,1:zlen,:,:]
    for i in range(zlen-1):
        grad[:,i,:,:] = dphi[:,i,:,:]/dz[i]
    return grad

def get_z_gradient(phi_w, z_w):
    t_axis, z_axis, y_axis, x_axis = 0,1,2,3
    tlen, zlen, ylen, xlen = phi_w.shape
    grad = np.zeros((tlen, zlen-1, ylen, xlen), dtype=np.float64)
    dz = np.roll(z_w,-1, axis=z_axis) - z_w
    dz = dz[:,0:-1,:,:]
    phi_plus = np.roll(phi_w,-1, axis=z_axis)
    dphi = phi_plus - phi_w
    dphi = dphi[:,0:-1,:,:]
    grad[:,:,:,:] = dphi[:,:,:,:]/dz[:,:,:,:]
    return grad


def calcAllThreeGradInCart(pm, pn, arr, s_w, dxi_zA, deta_zA, dsigma_zA):
    dxi_arr, deta_arr = gethorizontal_grad(pm, pn, arr)
    dsigma_arr = get_vertical_gradient(arr, s_w) 
    dz_arr = dsigma_arr / dsigma_zA
    dx_arr = dxi_arr - dxi_zA*dz_arr
    dy_arr = deta_arr - deta_zA*dz_arr
    return dx_arr, dy_arr, dz_arr