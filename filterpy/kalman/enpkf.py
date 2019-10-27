# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init

"""Copyright 2019 Valentin Noah Hartmann

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy
import numpy as np
from numpy import array, zeros, eye, dot
from numpy.random import multivariate_normal
from filterpy.common import pretty_str, outer_product_sum

from scipy.stats import multivariate_normal as mn

class EnsembleParticleKalmanFilter(object):
    """
    This method implements the EnsembleParticleKalmanFilter, a mixture
    of the Ensemble Kalman Filter, and the Particle Filter.

    The prediction of the state is the same as used in the Ensemble Kalman Filter,
    while the update is a mixture of a particle filter, and the Ensemble Kalman
    Filter. The weights of the mixture are chosen according to the parameter gamma.
    
    The details are described in [1].

    Parameters
    ----------

    x : np.array(dim_x)
        state mean

    P : np.array((dim_x, dim_x))
        covariance of the state

    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.

    dt : float
        time step in seconds

    N : int
        number of sigma points (ensembles). Must be greater than 1.

    K : np.array
        Kalman gain

    H : np.array
        Measurement function. Converts x into a measurement.

    fx : function fx(x, dt)
        State transition function. May be linear or nonlinear. Projects
        state x into the next time period. Returns the projected state x.


    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        State estimate

    P : numpy.array(dim_x, dim_x)
        State covariance matrix

    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.

    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.

    z : numpy.array
        Last measurement used in update(). Read only.

    R : numpy.array(dim_z, dim_z)
        Measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        Process noise matrix

    fx : callable (x, dt)
        State transition function

    H : np.array(dim_z, dim_x)
        Measurement matrix. Convert state `x` into a measurement

    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step. Read only.

    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv

    Examples
    --------

    .. code-block:: Python

        H = np.array([x[0]])
        F = np.array([[1., 1.],
                      [0., 1.]])
        def fx(x, dt):
            return np.dot(F, x)

        x = np.array([0., 1.])
        P = np.eye(2) * 100.
        dt = 0.1
        f = EnsembleParticleKalmanFilter(x=x, P=P, dim_z=1, dt=dt,
                                 N=8, H=H, fx=fx, gamma=0.5)

        std_noise = 3.
        f.R *= std_noise**2
        f.Q = Q_discrete_white_noise(2, dt, .01)

        while True:
            z = read_sensor()
            f.predict()
            f.update(np.asarray([z]))

    References
    ----------

    - [1] Frei, Marco, and Hans R. Künsch. "Bridging the ensemble Kalman and particle
          filters." Biometrika 100.4 (2013): 781-800.

    """

    def __init__(self, x, P, dim_z, dt, N, H, fx, gamma=.5):
        if dim_z <= 0:
            raise ValueError('dim_z must be greater than zero')

        if N <= 0:
            raise ValueError('N must be greater than zero')

        dim_x = len(x)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.N = N
        self.H = H
        self.fx = fx
        self.gamma = gamma

        if gamma == 0 or gamma == 1:
            raise ValueError()

        self.K = zeros((dim_x, dim_z))
        self.z = array([[None] * self.dim_z]).T
        self.S = zeros((dim_z, dim_z))   # system uncertainty
        self.SI = zeros((dim_z, dim_z))  # inverse system uncertainty

        self.initialize(x, P)
        self.Q = eye(dim_x)       # process uncertainty
        self.R = eye(dim_z)       # state uncertainty
        self.inv = np.linalg.inv

        # used to create error terms centered at 0 mean for
        # state and measurement
        self._mean = zeros(dim_x)
        self._mean_z = zeros(dim_z)

    def initialize(self, x, P):
        """
        Initializes the filter with the specified mean and
        covariance. Only need to call this if you are using the filter
        to filter more than one set of data; this is called by __init__

        Parameters
        ----------

        x : np.array(dim_z)
            state mean

        P : np.array((dim_x, dim_x))
            covariance of the state
        """

        if x.ndim != 1:
            raise ValueError('x must be a 1D array')

        self.sigmas = multivariate_normal(mean=x, cov=P, size=self.N)
        self.x = x
        self.P = P

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update(self, z, R=None):
        """
        Add a new measurement (z) to the kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------

        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise self.R will be used.
        """

        if z is None:
            self.z = array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if R is None:
            R = self.R
        if np.isscalar(R):
            R = eye(self.dim_z) * R

        H = self.H
        R = self.R

        N = self.N

        def K(gamma, L):
            PHT = np.dot(self.P, self.H.T)
            inv = self.inv(gamma*np.dot(H, PHT) + R)
            return gamma * np.dot(PHT, inv)

        def Q(gamma, K):
            return 1/gamma * np.dot(np.dot(K, R), K.T)

        K_gamma = K(self.gamma, self.P)
        nu = np.zeros_like(self.sigmas)
        for i, s in enumerate(self.sigmas):
            nu[i] = s + np.dot(K_gamma, (z - np.dot(H, s)))
        
        Q_gamma = Q(self.gamma, K_gamma)

        alpha = np.zeros((N))
        for i in range(N):
            alpha[i] = mn.pdf(z, np.dot(H, nu[i]), np.dot(np.dot(H, Q_gamma), H.T) + 1/(1-self.gamma) * R)

        norm = np.sum(alpha)

        indices = np.random.choice(np.arange(N), N, p=alpha / norm)
        
        e_1 = multivariate_normal(z*0, self.R, N)
        e_2 = multivariate_normal(z*0, self.R, N)

        x_gamma = np.zeros_like(nu)
        for i, ind in enumerate(indices):
            x_gamma[i] = nu[ind] + np.dot(K_gamma, e_1[i])/self.gamma**.5
        
        K_new = K((1 - self.gamma), Q(self.gamma, K_gamma))
        for i in range(N):
            diff = z + e_2[i] / (1 - self.gamma)**.5 - np.dot(H, x_gamma[i])
            self.sigmas[i] = x_gamma[i] + np.dot(K_new, diff) 

        self.x = np.mean(self.sigmas, axis=0)
        self.P = outer_product_sum(self.sigmas - self.x) / (N - 1)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self):
        """ Predict next position. """

        N = self.N
        for i, s in enumerate(self.sigmas):
            self.sigmas[i] = self.fx(s, self.dt)

        e = multivariate_normal(self._mean, self.Q, N)
        self.sigmas += e

        self.x = np.mean(self.sigmas, axis=0)
        self.P = outer_product_sum(self.sigmas - self.x) / (N - 1)

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def __repr__(self):
        return '\n'.join([
            'EnsembleKalmanFilter object',
            pretty_str('dim_x', self.dim_x),
            pretty_str('dim_z', self.dim_z),
            pretty_str('dt', self.dt),
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('K', self.K),
            pretty_str('S', self.S),
            pretty_str('sigmas', self.sigmas),
            pretty_str('H', self.H),
            pretty_str('fx', self.fx)
            ])
