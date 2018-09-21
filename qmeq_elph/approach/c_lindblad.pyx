"""Module containing cython functions, which generate first order Lindblad kernel.
   For docstrings see documentation of module lindblad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ..aprclass import Approach_elph
from ..specfuncc cimport Func_pauli_elph

from qmeq.mytypes import doublenp
from qmeq.mytypes import complexnp

from qmeq.approach.c_lindblad import c_generate_tLba
from qmeq.approach.c_lindblad import c_generate_kern_lindblad
from qmeq.approach.c_lindblad import c_generate_current_lindblad
from qmeq.approach.c_lindblad import c_generate_vec_lindblad

cimport numpy as np
cimport cython

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t

from libc.math cimport sqrt

#---------------------------------------------------------------------------------------------------------
# Lindblad approach
#---------------------------------------------------------------------------------------------------------
@cython.cdivision(True)
@cython.boundscheck(False)
def c_generate_tLbbp_elph(sys):
    cdef np.ndarray[complex_t, ndim=3] Vbbp = sys.baths.Vbbp
    cdef np.ndarray[double_t, ndim=1] E = sys.qd.Ea
    si = sys.si
    #
    cdef long_t b, bp
    cdef int_t  charge, l
    cdef double_t Ebbp
    cdef int_t nbaths = si.nbaths
    #
    cdef np.ndarray[complex_t, ndim=4] tLbbp = np.zeros((nbaths, si.nmany, si.nmany, 2), dtype=complexnp)
    #
    func_pauli = Func_pauli_elph(sys.baths.tlst, sys.baths.dlst,
                                 sys.baths.bath_func, sys.funcp.eps_elph)
    # Diagonal elements
    for l in range(nbaths):
        func_pauli.eval(0., l)
        for charge in range(si.ncharge):
            for b in si.statesdm[charge]:
                tLbbp[l, b, b, 0] = sqrt(0.5*func_pauli.val)*Vbbp[l, b, b]
                tLbbp[l, b, b, 1] = tLbbp[l, b, b, 0].conjugate()
    # Off-diagonal elements
    for charge in range(si.ncharge):
        for b, bp in itertools.permutations(si.statesdm[charge], 2):
            Ebbp = E[b]-E[bp]
            for l in range(nbaths):
                func_pauli.eval(Ebbp, l)
                tLbbp[l, b, bp, 0] = sqrt(0.5*func_pauli.val)*Vbbp[l, b, bp]
                tLbbp[l, b, bp, 1] = sqrt(0.5*func_pauli.val)*Vbbp[l, bp, b].conjugate()
    sys.tLbbp = tLbbp
    return 0

@cython.boundscheck(False)
def c_generate_kern_lindblad_elph(sys):
    cdef np.ndarray[double_t, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complex_t, ndim=4] tLbbp = sys.tLbbp
    si = sys.si
    cdef bint symq = sys.funcp.symq
    cdef long_t norm_rowp = sys.funcp.norm_row
    #
    cdef bool_t bbp_bool, bbpi_bool
    cdef int_t charge, l, q, nbaths, \
               aap_sgn, bppbp_sgn, bbpp_sgn
    cdef long_t b, bp, bbp, bbpi, bb, \
                a, ap, aap, aapi, \
                bpp, bppbp, bppbpi, bbpp, bbppi
    cdef long_t norm_row, last_row, ndm0, npauli,
    cdef complex_t fct_aap, fct_bppbp, fct_bbpp
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[bool_t, ndim=1] booldm0 = si.booldm0
    cdef np.ndarray[bool_t, ndim=1] conjdm0 = si.conjdm0
    #
    norm_row = norm_rowp if symq else si.ndm0r
    last_row = si.ndm0r-1 if symq else si.ndm0r
    ndm0, npauli, nbaths = si.ndm0, si.npauli, si.nbaths
    #
    cdef np.ndarray[double_t, ndim=2] kern
    if sys.kern is None:
        kern = np.zeros((last_row+1, si.ndm0r), dtype=doublenp)
    else:
        kern = sys.kern
    #
    for charge in range(si.ncharge):
        for b, bp in itertools.combinations_with_replacement(si.statesdm[charge], 2):
            bbp = mapdm0[lenlst[charge]*dictdm[b] + dictdm[bp] + shiftlst0[charge]]
            bbp_bool = booldm0[lenlst[charge]*dictdm[b] + dictdm[bp] + shiftlst0[charge]]
            if bbp != -1 and bbp_bool:
                bbpi = ndm0 + bbp - npauli
                bbpi_bool = True if bbpi >= ndm0 else False
                #--------------------------------------------------
                for a, ap in itertools.product(si.statesdm[charge], si.statesdm[charge]):
                    aap = mapdm0[lenlst[charge]*dictdm[a] + dictdm[ap] + shiftlst0[charge]]
                    if aap != -1:
                        fct_aap = 0
                        for l in range(nbaths):
                            for q in range(2):
                                fct_aap += tLbbp[l, b, a, q]*tLbbp[l, bp, a, q].conjugate()
                        aapi = ndm0 + aap - npauli
                        aap_sgn = +1 if conjdm0[lenlst[charge]*dictdm[a] + dictdm[ap] + shiftlst0[charge]] else -1
                        kern[bbp, aap] = kern[bbp, aap] + fct_aap.real                              # kern[bbp, aap]   += fct_aap.real
                        if aapi >= ndm0:
                            kern[bbp, aapi] = kern[bbp, aapi] - fct_aap.imag*aap_sgn                # kern[bbp, aapi]  -= fct_aap.imag*aap_sgn
                            if bbpi_bool:
                                kern[bbpi, aapi] = kern[bbpi, aapi] + fct_aap.real*aap_sgn          # kern[bbpi, aapi] += fct_aap.real*aap_sgn
                        if bbpi_bool:
                            kern[bbpi, aap] = kern[bbpi, aap] + fct_aap.imag                        # kern[bbpi, aap]  += fct_aap.imag
                #--------------------------------------------------
                for bpp in si.statesdm[charge]:
                    bppbp = mapdm0[lenlst[charge]*dictdm[bpp] + dictdm[bp] + shiftlst0[charge]]
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[charge]:
                            for l in range(nbaths):
                                for q in range(2):
                                    fct_bppbp += -0.5*tLbbp[l, a, b, q].conjugate()*tLbbp[l, a, bpp, q]
                        bppbpi = ndm0 + bppbp - npauli
                        bppbp_sgn = +1 if conjdm0[lenlst[charge]*dictdm[bpp] + dictdm[bp] + shiftlst0[charge]] else -1
                        kern[bbp, bppbp] = kern[bbp, bppbp] + fct_bppbp.real                        # kern[bbp, bppbp] += fct_bppbp.real
                        if bppbpi >= ndm0:
                            kern[bbp, bppbpi] = kern[bbp, bppbpi] - fct_bppbp.imag*bppbp_sgn        # kern[bbp, bppbpi] -= fct_bppbp.imag*bppbp_sgn
                            if bbpi_bool:
                                kern[bbpi, bppbpi] = kern[bbpi, bppbpi] + fct_bppbp.real*bppbp_sgn  # kern[bbpi, bppbpi] += fct_bppbp.real*bppbp_sgn
                        if bbpi_bool:
                            kern[bbpi, bppbp] = kern[bbpi, bppbp] + fct_bppbp.imag                  # kern[bbpi, bppbp] += fct_bppbp.imag
                    #--------------------------------------------------
                    bbpp = mapdm0[lenlst[charge]*dictdm[b] + dictdm[bpp] + shiftlst0[charge]]
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[charge]:
                            for l in range(nbaths):
                                for q in range(2):
                                    fct_bbpp += -0.5*tLbbp[l, a, bpp, q].conjugate()*tLbbp[l, a, bp, q]
                        bbppi = ndm0 + bbpp - npauli
                        bbpp_sgn = +1 if conjdm0[lenlst[charge]*dictdm[b] + dictdm[bpp] + shiftlst0[charge]] else -1
                        kern[bbp, bbpp] = kern[bbp, bbpp] + fct_bbpp.real                           # kern[bbp, bbpp] += fct_bbpp.real
                        if bbppi >= ndm0:
                            kern[bbp, bbppi] = kern[bbp, bbppi] - fct_bbpp.imag*bbpp_sgn            # kern[bbp, bbppi] -= fct_bbpp.imag*bbpp_sgn
                            if bbpi_bool:
                                kern[bbpi, bbppi] = kern[bbpi, bbppi] + fct_bbpp.real*bbpp_sgn      # kern[bbpi, bbppi] += fct_bbpp.real*bbpp_sgn
                        if bbpi_bool:
                            kern[bbpi, bbpp] = kern[bbpi, bbpp] + fct_bbpp.imag                     # kern[bbpi, bbpp] += fct_bbpp.imag
                #--------------------------------------------------
    # Normalisation condition
    kern[norm_row] = np.zeros(si.ndm0r, dtype=doublenp)
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = mapdm0[lenlst[charge]*dictdm[b] + dictdm[b] + shiftlst0[charge]]
            kern[norm_row, bb] += 1
    sys.kern = kern
    return 0

class Approach_Lindblad(Approach_elph):

    kerntype = 'Lindblad'
    generate_fct = c_generate_tLba
    generate_kern = c_generate_kern_lindblad
    generate_current = c_generate_current_lindblad
    generate_vec = c_generate_vec_lindblad
    #
    generate_kern_elph = c_generate_kern_lindblad_elph
    generate_fct_elph = c_generate_tLbbp_elph
#---------------------------------------------------------------------------------------------------------
