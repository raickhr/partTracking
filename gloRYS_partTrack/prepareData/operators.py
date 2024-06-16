import numpy as np

def gethorizontal_grad(dx, dy, phi):
    taxis = 0
    zaxis = 1
    yaxis = 2
    xaxis = 3
    grad_x = (np.roll(phi, -1, axis=xaxis) - np.roll(phi, 1, axis=xaxis))/(2*dx)
    grad_y = (np.roll(phi, -1, axis=yaxis) - np.roll(phi, 1, axis=yaxis))/(2*dy)
    return grad_x, grad_y

def gethorizontal_div(dx, dy, u, v, mask):
    taxis = 0
    zaxis = 1
    yaxis = 2
    xaxis = 3
    grad_x = (np.roll(u, -1, axis=xaxis) - np.roll(u, 1, axis=xaxis))/(2*dx)
    grad_y = (np.roll(v, -1, axis=yaxis) - np.roll(v, 1, axis=yaxis))/(2*dy)
    div = grad_x + grad_y
    div[mask] = 0
    return div