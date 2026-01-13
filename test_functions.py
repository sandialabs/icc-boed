import numpy as np


# 1D test problem function
def fun1D(x):
    y = x**2 - x * np.log(x + 1.) + (x - 0.5) * np.sin(2. * np.pi * x)
    return y


def grad1D(x):
    dy = 2. * x - np.log(x + 1.) - x / (x + 1.) + np.sin(2. * np.pi * x) \
        + (x - 0.5) * 2. * np.pi * np.cos(2. * np.pi * x)
    return dy


def hess1D(x):
    hess = 2. - 1./ (x + 1.) - 1. / (x + 1.)**2 \
        + 4. * np.pi * np.cos(2. * np.pi * x) \
        - (x - 0.5) * 4. * np.pi**2 * np.sin(2. * np.pi * x)
    return np.atleast_2d(hess)


# a second 1D test function
def fun1D_2(x):
    y = np.sin(x) + np.sin( 10/3 * x)
    return y


def grad1D_2(x):
    dy = np.cos(x) + (10/3) * np.cos( 10/3 * x)
    return dy


def hess1D_2(x):
    return - np.sin(x) - (10/3)**2 * np.sin( 10/3 * x)


# 2D test problem function
def fun2D(x):
    x = np.atleast_2d(x)
    assert x.shape[1] == 2
    return np.atleast_2d(x[:,0]**3 - 2*x[:,0]*x[:,1] - x[:,1]**6).T


def grad2D(x):
    x = np.atleast_2d(x)
    npoints = x.shape[0]
    grad_array = np.zeros((npoints,2))
    grad_array[:,0] = 3*x[:,0]**2 - 2*x[:,1]
    grad_array[:,1] = -2*x[:,0] - 6*x[:,1]**5
    return grad_array


def hess2D(x):
    x = np.atleast_2d(x)
    assert x.shape[0] == 1
    hess_mat = np.zeros((2, 2))
    hess_mat[0, 0] = 6. * x[:, 0]
    hess_mat[0, 1] = -2.
    hess_mat[1, 0] = -2.
    hess_mat[1, 1] = -30. * x[:, 1]**4
    return hess_mat


# a second 2D example
def fun2D_2(x):
    assert x.shape[1] == 2
    return np.atleast_2d(x[:,0]**2 + x[:,1]**2).T


def grad2D_2(x):
    x = np.atleast_2d(x)
    npoints = x.shape[0]
    grad_array = np.zeros((npoints, 2))
    grad_array[:,0] = 2*x[:,0]
    grad_array[:,1] = 2*x[:,1]
    return grad_array


def hess2D_2(x):
    x = np.atleast_2d(x)
    npoint = x.shape[0]
    hess_mat = np.zeros((npoints,2,2))
    hess_mat[:,0,0] = 2
    hess_mat[:,0,1] = 0
    hess_mat[:,1,0] = 0
    hess_mat[:,1,1] = 2
    return hess_mat


# 2D Rosenbrock equation
def Rosenbrock_2D(x):
    x = np.atleast_2d(x)
    y = 100. * (x[:, 1] - x[:, 0] ** 2) ** 2 + (x[:, 0] - 1) ** 2
    return y


def Rosenbrock_2D_grad(x):
    x = np.atleast_2d(x)
    npoints = x.shape[0]
    grad_array = np.zeros((npoints,2))
    grad_array[:, 0] = -400. * x[:, 0] * x[:, 1] + 400. * x[:, 0] ** 3 + 2 * x[:, 0] - 2
    grad_array[:, 0] = 200. * (x[:, 1] - x[:, 0] ** 2)
    return grad_array


def Rosenbrock_2D_hess(x):
    x = np.atleast_2d(x)
    assert x.shape[0] == 1
    hess_mat = np.zeros((2, 2))
    hess_mat[0, 0] = -400. * x[:, 1] + 1200. * x[:, 0] ** 2 + 2
    hess_mat[0, 1] = -400. * x[:, 0]
    hess_mat[1, 0] = -400. * x[:, 0]
    hess_mat[1, 1] = 200.
    return hess_mat


def sine_fun(theta, x):
    """
    y = sin(Bx-C)+D
 
    theta[0] = B
    theta[1] = C
    theta[2] = D

    """
    if len(theta.shape) > 1:
        theta = theta.squeeze()

    x = np.atleast_2d(x)
    y = np.sin(theta[0]*x[:,0] - theta[1])\
        + theta[2]
 
    return y


def sine_grad(theta, x):
    """
    dy/dx = Bcos(Bx-C)

    """

    x = np.atleast_2d(x)
    dx = theta[0]*cos(theta[0]*x[:,0] - theta[1])
    return dx


# 2D parametric test function
def fun2D_parametric(theta, x):

    assert x.shape[1] == 2
    assert theta.shape[1] == 2

    vals = x[:,0]**3 - theta[:, 0]*x[:,0]*x[:,1] - theta[:, 1]*x[:,1]**6
    return np.atleast_2d(vals).T


# DTS: should make this one more complicated
def fun3D(x):
    x = np.atleast_2d(x)
    return x[:,0]**3 + x[:,1]**3 + x[:,2]**3 - \
        9*x[:,0]*x[:,1] - 9*x[:,0]*x[:,2] + 27*x[:,0]


def grad3D(x):
    x = np.atleast_2d(x)
    npoints = x.shape[0]
    deriv_array = np.zeros((npoints,3))
    deriv_array[:,0] = 3*x[:,0]**2 - 9*x[:,1] - 9*x[:,2] + 27
    deriv_array[:,1] = 3*x[:,1]**2 - 9*x[:,0]
    deriv_array[:,2] = 3*x[:,2]**2 - 9*x[:,0]
    return deriv_array


# DTS: has some zero components
def hess3D(x):
    x = np.atleast_2d(x)
    hess_mat = np.zeros((3, 3))
    hess_mat[0,0] = 6*x[:,0]
    hess_mat[0,1] = -9
    hess_mat[0,2] = -9
    hess_mat[1,0] = -9
    hess_mat[1,1] = 6*x[:,1]
    hess_mat[1,2] = 0
    hess_mat[2,0] = -9
    hess_mat[2,1] = 0
    hess_mat[2,2] = 6*x[:,2]
    return hess_mat


# scaled 2D test function
# DTS: names are not consistent
#def fun2D_scaled(xx, trans=np.array([1., 1.]), scale=np.array([1., 1.])):

# DTS: function is symmetric w.r.t x and y
# makes it hard to see difference in performance
def fun_scaled_2D(unscaled_xx, scaler):
    if len(unscaled_xx.shape) == 1:
        unscaled_xx = np.atleast_2d(unscaled_xx)
    scaled_x = scaler.transform(unscaled_xx)
    x = scaled_x[:, 0]
    y = scaled_x[:, 1]
    return x**2 + y**3 + 2. * x * y

# DTS: returns the derivative in the unscaled space
def grad_scaled_2D(unscaled_xx, scaler):
    if len(unscaled_xx.shape) == 1:
       unscaled_xx = np.atleast_2d(unscaled_xx)
    scaled_x = scaler.transform(unscaled_xx)
    scale = scaler.scale_
    x = scaled_x[:, 0]
    y = scaled_x[:, 1]
    deriv_array = np.zeros_like(unscaled_xx)
    deriv_array[:, 0] = (2. * x + 2. * y) / scale[0]
    deriv_array[:, 1] = (3. * y**2 + 2. * x) / scale[1]
    return deriv_array

# DTS: returns the Hessian in the unscaled space
def hess_scaled_2D(unscaled_xx, scaler):
    if len(unscaled_xx.shape) == 1:
       unscaled_xx = np.atleast_2d(unscaled_xx)
    assert unscaled_xx.shape[0] == 1
    scaled_x = scaler.transform(unscaled_xx)

    scale = scaler.scale_
    x = scaled_x[:, 0]
    y = scaled_x[:, 1]

    hessian = np.zeros((2, 2))
    hessian[0, 0] = 2. / scale[0]**2
    hessian[0, 1] = 2. / (scale[0] * scale[1])
    hessian[1, 0] = hessian[0, 1]
    hessian[1, 1] = 6. * y / scale[1]**2

    return hessian
