"""Module containing python functions, which generate first order 1vN kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ..aprclass import Approach_elph
from ..specfunc import Func_1vN_elph

from qmeq.mytypes import complexnp
from qmeq.mytypes import doublenp

from qmeq.approach.neumann1 import generate_phi1fct
from qmeq.approach.neumann1 import generate_kern_1vN
from qmeq.approach.neumann1 import generate_current_1vN
from qmeq.approach.neumann1 import generate_vec_1vN

def generate_w1fct_elph(sys):
    (E, si) = (sys.qd.Ea, sys.si)
    w1fct = np.zeros((si.nbaths, si.ndm0, 2, 2), dtype=complexnp)
    func_1vN_elph = Func_1vN_elph(sys.baths.tlst, sys.baths.dlst,
                                  sys.funcp.itype_ph, sys.funcp.dqawc_limit,
                                  sys.baths.bath_func,
                                  sys.funcp.eps_elph)
    # Off-diagonal elements
    for l in range(si.nbaths):
        func_1vN_elph.eval(0., l)
        for charge in range(si.ncharge):
            for b in si.statesdm[charge]:
                bb = si.get_ind_dm0(b, b, charge)
                bb_bool = si.get_ind_dm0(b, b, charge, maptype=2)
                if bb != -1 and bb_bool:
                    w1fct[l, bb, 0, 1] = func_1vN_elph.val0 - 0.5j*func_1vN_elph.val0.imag
                    w1fct[l, bb, 1, 1] = func_1vN_elph.val1 - 0.5j*func_1vN_elph.val1.imag
    # Diagonal elements
    for charge in range(si.ncharge):
        for b, bp in itertools.combinations(si.statesdm[charge], 2):
            bbp = si.get_ind_dm0(b, bp, charge)
            bbp_bool = si.get_ind_dm0(b, bp, charge, maptype=2)
            if bbp != -1 and bbp_bool:
                Ebbp = E[b]-E[bp]
                for l in range(si.nbaths):
                    func_1vN_elph.eval(Ebbp, l)
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
def generate_kern_1vN_elph(sys):
    (E, Vbbp, w1fct, si, symq, norm_rowp) = (sys.qd.Ea, sys.baths.Vbbp, sys.w1fct, sys.si, sys.funcp.symq, sys.funcp.norm_row)
    norm_row = norm_rowp if symq else si.ndm0r
    last_row = si.ndm0r-1 if symq else si.ndm0r
    kern = sys.kern
    if kern is None:
        kern = np.zeros((last_row+1, si.ndm0r), dtype=doublenp)
    # Here letter convention is not used
    # For example, the label `a' has the same charge as the label `b'
    for charge in range(si.ncharge):
        for b, bp in itertools.combinations_with_replacement(si.statesdm[charge], 2):
            bbp = si.get_ind_dm0(b, bp, charge)
            bbp_bool = si.get_ind_dm0(b, bp, charge, 2)
            if bbp != -1 and bbp_bool:
                bbpi = si.ndm0 + bbp - si.npauli
                bbpi_bool = True if bbpi >= si.ndm0 else False
                #--------------------------------------------------
                for a, ap in itertools.product(si.statesdm[charge], si.statesdm[charge]):
                    aap = si.get_ind_dm0(a, ap, charge)
                    if aap != -1:
                        bpa = si.get_ind_dm0(bp, a, charge)
                        bap = si.get_ind_dm0(b, ap, charge)
                        bpa_conj = si.get_ind_dm0(bp, a, charge, maptype=3)
                        bap_conj = si.get_ind_dm0(b, ap, charge, maptype=3)
                        fct_aap = 0
                        for l in range(si.nbaths):
                            fct_aap += (+Vbbp[l, b, a]*Vbbp[l, ap, bp]*w1fct[l, bpa, 0, bpa_conj].conjugate()
                                        -Vbbp[l, b, a]*Vbbp[l, ap, bp]*w1fct[l, bap, 0, bap_conj])
                        aapi = si.ndm0 + aap - si.npauli
                        aap_sgn = +1 if si.get_ind_dm0(a, ap, charge, maptype=3) else -1
                        kern[bbp, aap] += fct_aap.imag                          # kern[bbp, aap]   += fct_aap.imag
                        if aapi >= si.ndm0:
                            kern[bbp, aapi] += fct_aap.real*aap_sgn             # kern[bbp, aapi]  += fct_aap.real*aap_sgn
                            if bbpi_bool:
                                kern[bbpi, aapi] += fct_aap.imag*aap_sgn        # kern[bbpi, aapi] += fct_aap.imag*aap_sgn
                        if bbpi_bool:
                            kern[bbpi, aap] -= fct_aap.real                     # kern[bbpi, aap]  -= fct_aap.real
                #--------------------------------------------------
                for bpp in si.statesdm[charge]:
                    bppbp = si.get_ind_dm0(bpp, bp, charge)
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[charge]:
                            bpa = si.get_ind_dm0(bp, a, charge)
                            bpa_conj = si.get_ind_dm0(bp, a, charge, maptype=3)
                            for l in range(si.nbaths):
                                fct_bppbp += +Vbbp[l, b, a]*Vbbp[l, a, bpp]*w1fct[l, bpa, 1, bpa_conj].conjugate()
                        for c in si.statesdm[charge]:
                            cbp = si.get_ind_dm0(c, bp, charge)
                            cbp_conj = si.get_ind_dm0(c, bp, charge, maptype=3)
                            for l in range(si.nbaths):
                                fct_bppbp += +Vbbp[l, b, c]*Vbbp[l, c, bpp]*w1fct[l, cbp, 0, cbp_conj]
                        bppbpi = si.ndm0 + bppbp - si.npauli
                        bppbp_sgn = +1 if si.get_ind_dm0(bpp, bp, charge, maptype=3) else -1
                        kern[bbp, bppbp] += fct_bppbp.imag                      # kern[bbp, bppbp] += fct_bppbp.imag
                        if bppbpi >= si.ndm0:
                            kern[bbp, bppbpi] += fct_bppbp.real*bppbp_sgn       # kern[bbp, bppbpi] += fct_bppbp.real*bppbp_sgn
                            if bbpi_bool:
                                kern[bbpi, bppbpi] += fct_bppbp.imag*bppbp_sgn  # kern[bbpi, bppbpi] += fct_bppbp.imag*bppbp_sgn
                        if bbpi_bool:
                            kern[bbpi, bppbp] -= fct_bppbp.real                 # kern[bbpi, bppbp] -= fct_bppbp.real
                    #--------------------------------------------------
                    bbpp = si.get_ind_dm0(b, bpp, charge)
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[charge]:
                            ba = si.get_ind_dm0(b, a, charge)
                            ba_conj = si.get_ind_dm0(b, a, charge, maptype=3)
                            for l in range(si.nbaths):
                                fct_bbpp += -Vbbp[l, bpp, a]*Vbbp[l, a, bp]*w1fct[l, ba, 1, ba_conj]
                        for c in si.statesdm[charge]:
                            cb = si.get_ind_dm0(c, b, charge)
                            cb_conj = si.get_ind_dm0(c, b, charge, maptype=3)
                            for l in range(si.nbaths):
                                fct_bbpp += -Vbbp[l, bpp, c]*Vbbp[l, c, bp]*w1fct[l, cb, 0, cb_conj].conjugate()
                        bbppi = si.ndm0 + bbpp - si.npauli
                        bbpp_sgn = +1 if si.get_ind_dm0(b, bpp, charge, maptype=3) else -1
                        kern[bbp, bbpp] += fct_bbpp.imag                        # kern[bbp, bbpp] += fct_bbpp.imag
                        if bbppi >= si.ndm0:
                            kern[bbp, bbppi] += fct_bbpp.real*bbpp_sgn          # kern[bbp, bbppi] += fct_bbpp.real*bbpp_sgn
                            if bbpi_bool:
                                kern[bbpi, bbppi] += fct_bbpp.imag*bbpp_sgn     # kern[bbpi, bbppi] += fct_bbpp.imag*bbpp_sgn
                        if bbpi_bool:
                            kern[bbpi, bbpp] -= fct_bbpp.real                   # kern[bbpi, bbpp] -= fct_bbpp.real
                #--------------------------------------------------
                for c, cp in itertools.product(si.statesdm[charge], si.statesdm[charge]):
                    ccp = si.get_ind_dm0(c, cp, charge)
                    if ccp != -1:
                        cbp = si.get_ind_dm0(c, bp, charge)
                        cpb = si.get_ind_dm0(cp, b, charge)
                        cbp_conj = si.get_ind_dm0(c, bp, charge, maptype=3)
                        cpb_conj = si.get_ind_dm0(cp, b, charge, maptype=3)
                        fct_ccp = 0
                        for l in range(si.nbaths):
                            fct_ccp += (+Vbbp[l, b, c]*Vbbp[l, cp, bp]*w1fct[l, cbp, 1, cbp_conj]
                                        -Vbbp[l, b, c]*Vbbp[l, cp, bp]*w1fct[l, cpb, 1, cpb_conj].conjugate())
                        ccpi = si.ndm0 + ccp - si.npauli
                        ccp_sgn = +1 if si.get_ind_dm0(c, cp, charge, maptype=3) else -1
                        kern[bbp, ccp] += fct_ccp.imag                          # kern[bbp, ccp] += fct_ccp.imag
                        if ccpi >= si.ndm0:
                            kern[bbp, ccpi] += fct_ccp.real*ccp_sgn             # kern[bbp, ccpi] += fct_ccp.real*ccp_sgn
                            if bbpi_bool:
                                kern[bbpi, ccpi] += fct_ccp.imag*ccp_sgn        # kern[bbpi, ccpi] += fct_ccp.imag*ccp_sgn
                        if bbpi_bool:
                            kern[bbpi, ccp] -= fct_ccp.real                     # kern[bbpi, ccp] -= fct_ccp.real
                #--------------------------------------------------
    # Normalisation condition
    kern[norm_row] = np.zeros(si.ndm0r, dtype=doublenp)
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = si.get_ind_dm0(b, b, charge)
            kern[norm_row, bb] += 1
    sys.kern = kern
    return 0

class Approach_py1vN(Approach_elph):

    kerntype = 'py1vN'
    generate_fct = staticmethod(generate_phi1fct)
    generate_kern = staticmethod(generate_kern_1vN)
    generate_current = staticmethod(generate_current_1vN)
    generate_vec = staticmethod(generate_vec_1vN)
    #
    generate_kern_elph = staticmethod(generate_kern_1vN_elph)
    generate_fct_elph = staticmethod(generate_w1fct_elph)
#---------------------------------------------------------------------------------------------------------
