"""Module containing cython functions, which generate first order 1vN kernel.
   For docstrings see documentation of module neumann1."""

# cython: profile=True

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ..aprclass import Approach_elph
from ..specfuncc cimport Func_1vN_elph

from qmeq.mytypes import doublenp
from qmeq.mytypes import complexnp

from qmeq.approach.c_neumann1 import c_generate_phi1fct
from qmeq.approach.c_neumann1 import c_generate_kern_1vN
from qmeq.approach.c_neumann1 import c_generate_current_1vN
from qmeq.approach.c_neumann1 import c_generate_vec_1vN

cimport numpy as np
cimport cython

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t

@cython.boundscheck(False)
def c_generate_w1fct_elph(sys):
    cdef np.ndarray[double_t, ndim=1] E = sys.qd.Ea
    si = sys.si
    #
    cdef bool_t bbp_bool, bb_bool
    cdef long_t b, bp, bbp, bb
    cdef int_t charge, l
    cdef double_t Ebbp
    #
    cdef int_t nbaths = si.nbaths
    #
    cdef np.ndarray[complex_t, ndim=4] w1fct = np.zeros((nbaths, si.ndm0, 2, 2), dtype=complexnp)
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[bool_t, ndim=1] booldm0 = si.booldm0
    #
    func_1vN_elph = Func_1vN_elph(sys.baths.tlst, sys.baths.dlst,
                                  sys.funcp.itype_ph, sys.funcp.dqawc_limit,
                                  sys.baths.bath_func,
                                  sys.funcp.eps_elph)
    # Diagonal elements
    for l in range(nbaths):
        func_1vN_elph.eval(0., l)
        for charge in range(si.ncharge):
            for b in si.statesdm[charge]:
                bb = mapdm0[lenlst[charge]*dictdm[b] + dictdm[b] + shiftlst0[charge]]
                bb_bool = booldm0[lenlst[charge]*dictdm[b] + dictdm[b] + shiftlst0[charge]]
                if bb != -1 and bb_bool:
                    w1fct[l, bb, 0, 1] = func_1vN_elph.val0 - 0.5j*func_1vN_elph.val0.imag
                    w1fct[l, bb, 1, 1] = func_1vN_elph.val1 - 0.5j*func_1vN_elph.val1.imag
    # Off-diagonal elements
    for charge in range(si.ncharge):
        for b, bp in itertools.combinations(si.statesdm[charge], 2):
            bbp = mapdm0[lenlst[charge]*dictdm[b] + dictdm[bp] + shiftlst0[charge]]
            bbp_bool = booldm0[lenlst[charge]*dictdm[b] + dictdm[bp] + shiftlst0[charge]]
            if bbp != -1 and bbp_bool:
                Ebbp = E[b]-E[bp]
                for l in range(nbaths):
                    func_1vN_elph.eval(Ebbp, l)
                    val = func_1vN_elph.val0
                    w1fct[l, bbp, 0, 1] = func_1vN_elph.val0
                    w1fct[l, bbp, 1, 1] = func_1vN_elph.val1
                    func_1vN_elph.eval(-Ebbp, l)
                    w1fct[l, bbp, 0, 0] = func_1vN_elph.val0
                    w1fct[l, bbp, 1, 0] = func_1vN_elph.val1
    sys.w1fct = w1fct
    return 0

#---------------------------------------------------------------------------------------------------------
# 1 von Neumann approach
#---------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
def c_generate_kern_1vN_elph(sys):
    cdef np.ndarray[double_t, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complex_t, ndim=3] Vbbp = sys.baths.Vbbp
    cdef np.ndarray[complex_t, ndim=4] w1fct = sys.w1fct
    si = sys.si
    cdef bint symq = sys.funcp.symq
    cdef long_t norm_rowp = sys.funcp.norm_row
    #
    cdef bool_t bbp_bool, bbpi_bool, \
                bpa_conj, bap_conj, cbp_conj, \
                ba_conj, cb_conj, cpb_conj
    cdef int_t charge, l, nbaths, \
               aap_sgn, bppbp_sgn, bbpp_sgn, ccp_sgn
    cdef long_t b, bp, bbp, bbpi, bb, \
                a, ap, aap, aapi, \
                bpp, bppbp, bppbpi, bbpp, bbppi, \
                c, cp, ccp, ccpi, \
                bpa, bap, cbp, ba, cb, cpb
    cdef long_t norm_row, last_row, ndm0, npauli,
    cdef complex_t fct_aap, fct_bppbp, fct_bbpp, fct_ccp
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
                        bpa = mapdm0[lenlst[charge]*dictdm[bp] + dictdm[a] + shiftlst0[charge]]
                        bap = mapdm0[lenlst[charge]*dictdm[b] + dictdm[ap] + shiftlst0[charge]]
                        bpa_conj = conjdm0[lenlst[charge]*dictdm[bp] + dictdm[a] + shiftlst0[charge]]
                        bap_conj = conjdm0[lenlst[charge]*dictdm[b] + dictdm[ap] + shiftlst0[charge]]
                        fct_aap = 0
                        for l in range(nbaths):
                            fct_aap += (+Vbbp[l, b, a]*Vbbp[l, ap, bp]*w1fct[l, bpa, 0, bpa_conj].conjugate()
                                        -Vbbp[l, b, a]*Vbbp[l, ap, bp]*w1fct[l, bap, 0, bap_conj])
                        aapi = ndm0 + aap - npauli
                        aap_sgn = +1 if conjdm0[lenlst[charge]*dictdm[a] + dictdm[ap] + shiftlst0[charge]] else -1
                        kern[bbp, aap] = kern[bbp, aap] + fct_aap.imag                              # kern[bbp, aap]   += fct_aap.imag
                        if aapi >= ndm0:
                            kern[bbp, aapi] = kern[bbp, aapi] + fct_aap.real*aap_sgn                # kern[bbp, aapi]  += fct_aap.real*aap_sgn
                            if bbpi_bool:
                                kern[bbpi, aapi] = kern[bbpi, aapi] + fct_aap.imag*aap_sgn          # kern[bbpi, aapi] += fct_aap.imag*aap_sgn
                        if bbpi_bool:
                            kern[bbpi, aap] = kern[bbpi, aap] - fct_aap.real                        # kern[bbpi, aap]  -= fct_aap.real
                #--------------------------------------------------
                for bpp in si.statesdm[charge]:
                    bppbp = mapdm0[lenlst[charge]*dictdm[bpp] + dictdm[bp] + shiftlst0[charge]]
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[charge]:
                            bpa = mapdm0[lenlst[charge]*dictdm[bp] + dictdm[a] + shiftlst0[charge]]
                            bpa_conj = conjdm0[lenlst[charge]*dictdm[bp] + dictdm[a] + shiftlst0[charge]]
                            for l in range(nbaths):
                                fct_bppbp += +Vbbp[l, b, a]*Vbbp[l, a, bpp]*w1fct[l, bpa, 1, bpa_conj].conjugate()
                        for c in si.statesdm[charge]:
                            cbp = mapdm0[lenlst[charge]*dictdm[c] + dictdm[bp] + shiftlst0[charge]]
                            cbp_conj = conjdm0[lenlst[charge]*dictdm[c] + dictdm[bp] + shiftlst0[charge]]
                            for l in range(nbaths):
                                fct_bppbp += +Vbbp[l, b, c]*Vbbp[l, c, bpp]*w1fct[l, cbp, 0, cbp_conj]
                        bppbpi = ndm0 + bppbp - npauli
                        bppbp_sgn = +1 if conjdm0[lenlst[charge]*dictdm[bpp] + dictdm[bp] + shiftlst0[charge]] else -1
                        kern[bbp, bppbp] = kern[bbp, bppbp] + fct_bppbp.imag                        # kern[bbp, bppbp] += fct_bppbp.imag
                        if bppbpi >= ndm0:
                            kern[bbp, bppbpi] = kern[bbp, bppbpi] + fct_bppbp.real*bppbp_sgn        # kern[bbp, bppbpi] += fct_bppbp.real*bppbp_sgn
                            if bbpi_bool:
                                kern[bbpi, bppbpi] = kern[bbpi, bppbpi] + fct_bppbp.imag*bppbp_sgn  # kern[bbpi, bppbpi] += fct_bppbp.imag*bppbp_sgn
                        if bbpi_bool:
                            kern[bbpi, bppbp] = kern[bbpi, bppbp] - fct_bppbp.real                  # kern[bbpi, bppbp] -= fct_bppbp.real
                    #--------------------------------------------------
                    bbpp = mapdm0[lenlst[charge]*dictdm[b] + dictdm[bpp] + shiftlst0[charge]]
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[charge]:
                            ba = mapdm0[lenlst[charge]*dictdm[b] + dictdm[a] + shiftlst0[charge]]
                            ba_conj = conjdm0[lenlst[charge]*dictdm[b] + dictdm[a] + shiftlst0[charge]]
                            for l in range(nbaths):
                                fct_bbpp += -Vbbp[l, bpp, a]*Vbbp[l, a, bp]*w1fct[l, ba, 1, ba_conj]
                        for c in si.statesdm[charge]:
                            cb = mapdm0[lenlst[charge]*dictdm[c] + dictdm[b] + shiftlst0[charge]]
                            cb_conj = conjdm0[lenlst[charge]*dictdm[c] + dictdm[b] + shiftlst0[charge]]
                            for l in range(nbaths):
                                fct_bbpp += -Vbbp[l, bpp, c]*Vbbp[l, c, bp]*w1fct[l, cb, 0, cb_conj].conjugate()
                        bbppi = ndm0 + bbpp - npauli
                        bbpp_sgn = +1 if conjdm0[lenlst[charge]*dictdm[b] + dictdm[bpp] + shiftlst0[charge]] else -1
                        kern[bbp, bbpp] = kern[bbp, bbpp] + fct_bbpp.imag                           # kern[bbp, bbpp] += fct_bbpp.imag
                        if bbppi >= ndm0:
                            kern[bbp, bbppi] = kern[bbp, bbppi] + fct_bbpp.real*bbpp_sgn            # kern[bbp, bbppi] += fct_bbpp.real*bbpp_sgn
                            if bbpi_bool:
                                kern[bbpi, bbppi] = kern[bbpi, bbppi] + fct_bbpp.imag*bbpp_sgn      # kern[bbpi, bbppi] += fct_bbpp.imag*bbpp_sgn
                        if bbpi_bool:
                            kern[bbpi, bbpp] = kern[bbpi, bbpp] - fct_bbpp.real                     # kern[bbpi, bbpp] -= fct_bbpp.real
                #--------------------------------------------------
                for c, cp in itertools.product(si.statesdm[charge], si.statesdm[charge]):
                    ccp = mapdm0[lenlst[charge]*dictdm[c] + dictdm[cp] + shiftlst0[charge]]
                    if ccp != -1:
                        cbp = mapdm0[lenlst[charge]*dictdm[c] + dictdm[bp] + shiftlst0[charge]]
                        cpb = mapdm0[lenlst[charge]*dictdm[cp] + dictdm[b] + shiftlst0[charge]]
                        cbp_conj = conjdm0[lenlst[charge]*dictdm[c] + dictdm[bp] + shiftlst0[charge]]
                        cpb_conj = conjdm0[lenlst[charge]*dictdm[cp] + dictdm[b] + shiftlst0[charge]]
                        fct_ccp = 0
                        for l in range(nbaths):
                            fct_ccp += (+Vbbp[l, b, c]*Vbbp[l, cp, bp]*w1fct[l, cbp, 1, cbp_conj]
                                        -Vbbp[l, b, c]*Vbbp[l, cp, bp]*w1fct[l, cpb, 1, cpb_conj].conjugate())
                        ccpi = ndm0 + ccp - npauli
                        ccp_sgn = +1 if conjdm0[lenlst[charge]*dictdm[c] + dictdm[cp] + shiftlst0[charge]] else -1
                        kern[bbp, ccp] = kern[bbp, ccp] + fct_ccp.imag                              # kern[bbp, ccp] += fct_ccp.imag
                        if ccpi >= ndm0:
                            kern[bbp, ccpi] = kern[bbp, ccpi] + fct_ccp.real*ccp_sgn                # kern[bbp, ccpi] += fct_ccp.real*ccp_sgn
                            if bbpi_bool:
                                kern[bbpi, ccpi] = kern[bbpi, ccpi] + fct_ccp.imag*ccp_sgn          # kern[bbpi, ccpi] += fct_ccp.imag*ccp_sgn
                        if bbpi_bool:
                            kern[bbpi, ccp] = kern[bbpi, ccp] - fct_ccp.real                        # kern[bbpi, ccp] -= fct_ccp.real
                #--------------------------------------------------
    # Normalisation condition
    kern[norm_row] = np.zeros(si.ndm0r, dtype=doublenp)
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = mapdm0[lenlst[charge]*dictdm[b] + dictdm[b] + shiftlst0[charge]]
            kern[norm_row, bb] += 1
    sys.kern = kern
    return 0

class Approach_1vN(Approach_elph):

    kerntype = '1vN'
    generate_fct = c_generate_phi1fct
    generate_kern = c_generate_kern_1vN
    generate_current = c_generate_current_1vN
    generate_vec = c_generate_vec_1vN
    #
    generate_kern_elph = c_generate_kern_1vN_elph
    generate_fct_elph = c_generate_w1fct_elph
#---------------------------------------------------------------------------------------------------------
