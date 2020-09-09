# This is a Python module for implementing some useful classes partial differential equations
import numpy as np
import utility as ut

class QuasiLinearPDE0(object):
    """
    Implements Fokker-Planck type quasi-linear parabolic PDEs
    u_t + Lu = 0, (t, x) in a box domain
    u(0, x) = initial condition
    u(t, x) = g(t, x) boundary condition at the boundary of the space domain
    """
    def __init__(self, diff_op, init_cond, bdry_cond, space_domain, time_domain):
        """
        Description: Constructor for QuasiLinearPDE0
        Args:   diff_op: differential operator L
                init_cond: initial condition u(0, x)
                bdry_cond: boundary condition u(t, x) at the boundary of the space domain
                space_domain: box doamin for space in form of a dx2 matrix, d = space dimension
                time_domain: domain of time as a list/tuple/np.array [a, b]
        """
        self.diff_op = diff_op
        self.init_cond = init_cond
        self.bdry_cond = bdry_cond
        self.space_domain = np.array(space_domain)
        self.time_domain = time_domain
        self.space_dim = self.space_domain.shape[0]
        self.dim = self.space_dim + 1

    @ut.timer
    def domain_sampler(self, num_samples):
        """
        Description: sampling function for space-time domain
        Args: number of samples to generate
        Returns: an np.array where each row each is a singular sample from the space-time domain with the last coordinate being time
        """
        self.samples = np.zeros((num_samples, self.space_dim + 1))
        for j in range(self.space_dim):
            a, b = self.space_domain[j]
            self.samples[:, j] = (b - a)*np.random.random(num_samples) + a
        a, b = self.time_domain
        self.samples[:, self.space_dim] = (b - a)*np.random.random(num_samples) + a
        return self.samples

    def loss(self, func, input):
        """
        Description: loss function for solving the PDE with DGM
        """
        return self.diff_op(func, input)**2 + self.init_cond(input)**2 + self.bdry_cond(input)**2


class ODE0(object):
    """
    Implements ODEs with initial condition
    Lu = 0, x in a box domain
    u(0) = initial condition
    """
    def __init__(self, diff_op, init_cond, domain):
        """
        Description: Constructor for ODE0
        Args:   diff_op: differential operator L
                init_cond: initial condition u(0, x)
                domain: domain of x as a list/tuple/np.array [a, b]
        """
        self.diff_op = diff_op
        self.init_cond = init_cond
        self.domain = domain
        self.dim = 1

    @ut.timer
    def domain_sampler(self, num_samples):
        """
        Description: sampling function for space-time domain
        Args: number of samples to generate
        Returns: an np.array where each row each is a singular sample from the space-time domain with the last coordinate being time
        """
        self.samples = np.zeros((num_samples, self.dim))
        a, b = self.domain
        self.samples[:, 0] = (b - a)*np.random.random(num_samples) + a
        return self.samples

    def loss(self, func, input):
        """
        Description: loss function for solving the PDE with DGM
        """
        return self.diff_op(func, input)**2 + self.init_cond(input)**2
