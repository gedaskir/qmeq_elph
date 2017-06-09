"""Module containing python functions, which generate first order Linblad kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from qmeq.mytypes import complexnp
from qmeq.mytypes import doublenp
from ..specfunc import Func_pauli_elph


#---------------------------------------------------------------------------------------------------------
# Lindblad approach
#---------------------------------------------------------------------------------------------------------
def elph_generate_tLbbp(sys):
    (Vbbp, E, si) = (sys.baths.Vbbp, sys.qd.Ea, sys.si)
    mtype = sys.baths.mtype
    func_pauli = Func_pauli_elph(sys.baths.tlst, sys.baths.dlst,
                                 sys.baths.bath_func, sys.funcp.eps_elph)
    #
    tLbbp = np.zeros(Vbbp.shape, dtype=mtype)
    # Diagonal elements
    for l in range(si.nbaths):
        func_pauli.eval(0., l)
        for charge in range(si.ncharge):
            for b in si.statesdm[charge]:
                tLbbp[l, b, b] = np.sqrt(func_pauli.val)*Vbbp[l, b, b]
    # Off-diagonal elements
    for charge in range(si.ncharge):
        for b, bp in itertools.permutations(si.statesdm[charge], 2):
            Ebbp = E[b]-E[bp]
            for l in range(si.nbaths):
                func_pauli.eval(Ebbp, l)
                tLbbp[l, b, bp] = np.sqrt(func_pauli.val)*Vbbp[l, b, bp]
    return tLbbp

def elph_generate_kern_lindblad(sys):
    (kern, E, tLbbp) = (sys.kern, sys.qd.Ea, sys.tLbbp)
    (si, symq, norm_rowp) = (sys.si, sys.funcp.symq, sys.funcp.norm_row)
    norm_row = norm_rowp if symq else si.ndm0r
    last_row = si.ndm0r-1 if symq else si.ndm0r
    if kern is None:
        kern = np.zeros((last_row+1, si.ndm0r), dtype=doublenp)
    for charge in range(si.ncharge):
        for b, bp in itertools.combinations_with_replacement(si.statesdm[charge], 2):
            bbp = si.get_ind_dm0(b, bp, charge)
            bbp_bool = si.get_ind_dm0(b, bp, charge, 2)
            if bbp != -1 and bbp_bool:
                bbpi = si.ndm0 + bbp - si.npauli
                bbpi_bool = True if bbpi >= si.ndm0 else False
                #--------------------------------------------------
                # Here letter convention is not used
                # For example, the label `a' has the same charge as the label `b'
                for a, ap in itertools.product(si.statesdm[charge], si.statesdm[charge]):
                    aap = si.get_ind_dm0(a, ap, charge)
                    if aap != -1:
                        fct_aap = 0
                        for l in range(si.nbaths):
                            fct_aap += tLbbp[l, b, a]*tLbbp[l, bp, ap].conjugate()
                        aapi = si.ndm0 + aap - si.npauli
                        aap_sgn = +1 if si.get_ind_dm0(a, ap, charge, maptype=3) else -1
                        kern[bbp, aap] += fct_aap.real                          # kern[bbp, aap]   += fct_aap.real
                        if aapi >= si.ndm0:
                            kern[bbp, aapi] -= fct_aap.imag*aap_sgn             # kern[bbp, aapi]  -= fct_aap.imag*aap_sgn
                            if bbpi_bool:
                                kern[bbpi, aapi] += fct_aap.real*aap_sgn        # kern[bbpi, aapi] += fct_aap.real*aap_sgn
                        if bbpi_bool:
                            kern[bbpi, aap] += fct_aap.imag                     # kern[bbpi, aap]  += fct_aap.imag
                #--------------------------------------------------
                for bpp in si.statesdm[charge]:
                    bppbp = si.get_ind_dm0(bpp, bp, charge)
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[charge]:
                            for l in range(si.nbaths):
                                fct_bppbp += -0.5*tLbbp[l, a, b].conjugate()*tLbbp[l, a, bpp]
                        bppbpi = si.ndm0 + bppbp - si.npauli
                        bppbp_sgn = +1 if si.get_ind_dm0(bpp, bp, charge, maptype=3) else -1
                        kern[bbp, bppbp] += fct_bppbp.real                      # kern[bbp, bppbp] += fct_bppbp.real
                        if bppbpi >= si.ndm0:
                            kern[bbp, bppbpi] -= fct_bppbp.imag*bppbp_sgn       # kern[bbp, bppbpi] -= fct_bppbp.imag*bppbp_sgn
                            if bbpi_bool:
                                kern[bbpi, bppbpi] += fct_bppbp.real*bppbp_sgn  # kern[bbpi, bppbpi] += fct_bppbp.real*bppbp_sgn
                        if bbpi_bool:
                            kern[bbpi, bppbp] += fct_bppbp.imag                 # kern[bbpi, bppbp] += fct_bppbp.imag
                    #--------------------------------------------------
                    bbpp = si.get_ind_dm0(b, bpp, charge)
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[charge]:
                            for l in range(si.nbaths):
                                fct_bbpp += -0.5*tLbbp[l, a, bpp].conjugate()*tLbbp[l, a, bp]
                        bbppi = si.ndm0 + bbpp - si.npauli
                        bbpp_sgn = +1 if si.get_ind_dm0(b, bpp, charge, maptype=3) else -1
                        kern[bbp, bbpp] += fct_bbpp.real                        # kern[bbp, bbpp] += fct_bbpp.real
                        if bbppi >= si.ndm0:
                            kern[bbp, bbppi] -= fct_bbpp.imag*bbpp_sgn          # kern[bbp, bbppi] -= fct_bbpp.imag*bbpp_sgn
                            if bbpi_bool:
                                kern[bbpi, bbppi] += fct_bbpp.real*bbpp_sgn     # kern[bbpi, bbppi] += fct_bbpp.real*bbpp_sgn
                        if bbpi_bool:
                            kern[bbpi, bbpp] += fct_bbpp.imag                   # kern[bbpi, bbpp] += fct_bbpp.imag
                #--------------------------------------------------
    # Normalisation condition
    kern[norm_row] = np.zeros(si.ndm0r, dtype=doublenp)
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = si.get_ind_dm0(b, b, charge)
            kern[norm_row, bb] += 1
    return kern
#---------------------------------------------------------------------------------------------------------
