"""Module containing various special functions, cython implementation."""

# cython: profile=True

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.integrate import quad

from qmeq.mytypes import doublenp
from qmeq.mytypes import complexnp

cimport numpy as np
cimport cython

cdef double_t pi = 3.14159265358979323846
from libc.math cimport exp
from libc.math cimport log

#-----------------------------------------------------------------------
cdef class Func:
    cpdef double_t eval(self, double_t x):
        return 1.

@cython.cdivision(True)
cdef class Func_bose:
    """Bose function."""
    cdef double_t eval(self, double_t x):
        return 1./(exp(x)-1.)

@cython.cdivision(True)
@cython.boundscheck(False)
cdef class Func_pauli_elph:

    def __init__(self, np.ndarray[double_t, ndim=1] tlst,
                       np.ndarray[double_t, ndim=2] dlst,
                       bath_func,
                       double_t eps):
        self.tlst, self.dlst = tlst, dlst
        self.bath_func = bath_func
        self.bath_func_q = False if bath_func is None else True
        self.eps = eps
        #
        self.bose, self.dos = Func_bose(), Func()
        self.val = 0.

    cpdef void eval(self, double_t Ebbp, int_t l):
        cdef double_t T, omm, omp
        cdef double_t alpha, Rm, Rp
        T, omm, omp = self.tlst[l], self.dlst[l,0], self.dlst[l,1]
        #alpha, Rm, Rp = Ebbp/T, omm/T, omp/T
        alpha = max(abs(Ebbp/T), self.eps) * (1 if Ebbp >= 0 else -1)
        Rm, Rp = max(omm/T, 0.9*self.eps), omp/T
        if self.bath_func_q:
            self.dos = self.bath_func[l]
        if alpha < Rp and alpha > Rm:
            # Absorption
            self.val = 2*pi*self.bose.eval(alpha)*self.dos.eval(T*alpha)
        elif -alpha < Rp and -alpha > Rm:
            # Emission
            self.val = 2*pi*(1+self.bose.eval(-alpha))*self.dos.eval(-T*alpha)
        else:
            self.val = 0.

@cython.cdivision(True)
@cython.boundscheck(False)
cdef class Func_1vN_elph:

    def __init__(self, np.ndarray[double_t, ndim=1] tlst,
                       np.ndarray[double_t, ndim=2] dlst,
                       int_t itype,
                       long_t limit,
                       bath_func,
                       double_t eps):
        self.tlst, self.dlst = tlst, dlst
        self.itype, self.limit = itype, limit
        self.bath_func = bath_func
        self.bath_func_q = False if bath_func is None else True
        self.eps = eps
        #
        self.bose, self.dos = Func_bose(), Func()
        self.val0, self.val1 = 0., 0.

    cpdef double_t iplus(self, double_t x):
        return +self.dos.eval(self.T*x)*self.bose.eval(x)

    cpdef double_t iminus(self, double_t x):
        return -self.dos.eval(self.T*x)*(1.+self.bose.eval(x))

    cpdef void eval(self, double_t Ebbp, int_t l):
        cdef double_t T, omm, omp
        cdef double_t alpha, Rm, Rp, err
        cdef complex_t val0, val1
        T, omm, omp = self.tlst[l], self.dlst[l,0], self.dlst[l,1]
        #alpha, Rm, Rp = Ebbp/T, omm/T, omp/T
        alpha = max(abs(Ebbp/T), self.eps) * (1 if Ebbp >= 0 else -1)
        Rm, Rp = max(omm/T, 0.9*self.eps), omp/T
        self.T = T
        if self.bath_func_q:
            self.dos = self.bath_func[l]
        if self.itype is 0:
            self.val0, err = quad(self.iplus, Rm, Rp, weight='cauchy', wvar=alpha,
                                  epsabs=1.0e-6, epsrel=1.0e-6, limit=self.limit)
            self.val0 = self.val0 + (-1.0j*pi*self.iplus(alpha) if alpha < Rp and alpha > Rm else 0)
            self.val1, err = quad(self.iminus, Rm, Rp, weight='cauchy', wvar=alpha,
                                  epsabs=1.0e-6, epsrel=1.0e-6, limit=self.limit)
            self.val1 = self.val1 + (-1.0j*pi*self.iminus(alpha) if alpha < Rp and alpha > Rm else 0)
        elif self.itype is 2:
            self.val0 = -1.0j*pi*self.iplus(alpha) if alpha < Rp and alpha > Rm else 0
            self.val1 = -1.0j*pi*self.iminus(alpha) if alpha < Rp and alpha > Rm else 0
#-----------------------------------------------------------------------

def main(int_t xpnt, func=None):
    import timeit
    tlst = np.array([1.], dtype=doublenp)
    dlst = np.array([[0.1,10.]], dtype=doublenp)
    f = Func_pauli_elph(tlst, dlst, func)

    cdef int_t i
    cdef np.ndarray[double_t, ndim=1] xlst = np.linspace(0.5,2.,xpnt)

    start = timeit.default_timer()
    for i in range(xpnt):
        f.eval(xlst[i], 0)
    stop = timeit.default_timer()
    print(stop-start)

def main2(int_t xpnt, func=None, itype=0):
    import timeit
    tlst = np.array([1.], dtype=doublenp)
    dlst = np.array([[0.1,10.]], dtype=doublenp)
    cdef np.ndarray[complex_t, ndim=1] rez = np.zeros(2, dtype=complexnp)
    f = Func_1vN_elph(tlst, dlst, itype, 10000, func)

    cdef int_t i
    cdef np.ndarray[double_t, ndim=1] xlst = np.linspace(0.5,2.,xpnt)
    cdef complex_t val0, val1

    start = timeit.default_timer()
    for i in range(xpnt):
        #f.eval(xlst[i], 0, rez)
        f.eval(xlst[i], 0)
        val0, val1 = f.val0, f.val1
    stop = timeit.default_timer()
    print(stop-start)
