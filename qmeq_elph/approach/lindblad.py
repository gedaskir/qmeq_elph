"""Module containing python functions, which generate first order Linblad kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ..aprclass import Approach_elph
from ..specfunc import Func_pauli_elph

from qmeq.mytypes import complexnp
from qmeq.mytypes import doublenp

from qmeq.approach.lindblad import generate_tLba
from qmeq.approach.lindblad import generate_kern_lindblad
from qmeq.approach.lindblad import generate_current_lindblad
from qmeq.approach.lindblad import generate_vec_lindblad

#---------------------------------------------------------------------------------------------------------
# Lindblad approach
#---------------------------------------------------------------------------------------------------------
def generate_tLbbp_elph(sys):
    (Vbbp, E, si) = (sys.baths.Vbbp, sys.qd.Ea, sys.si)
    mtype = sys.baths.mtype
    func_pauli = Func_pauli_elph(sys.baths.tlst, sys.baths.dlst,
                                 sys.baths.bath_func, sys.funcp.eps_elph)
    #
    tLbbp_shape = Vbbp.shape + (2,)
    tLbbp = np.zeros(tLbbp_shape, dtype=mtype)
    # Diagonal elements
    for l in range(si.nbaths):
        func_pauli.eval(0., l)
        for charge in range(si.ncharge):
            for b in si.statesdm[charge]:
                tLbbp[l, b, b, 0] = np.sqrt(0.5*func_pauli.val)*Vbbp[l, b, b]
                tLbbp[l, b, b, 1] = tLbbp[l, b, b, 0].conjugate()
    # Off-diagonal elements
    for charge in range(si.ncharge):
        for b, bp in itertools.permutations(si.statesdm[charge], 2):
            Ebbp = E[b]-E[bp]
            for l in range(si.nbaths):
                func_pauli.eval(Ebbp, l)
                tLbbp[l, b, bp, 0] = np.sqrt(0.5*func_pauli.val)*Vbbp[l, b, bp]
                tLbbp[l, b, bp, 1] = np.sqrt(0.5*func_pauli.val)*Vbbp[l, bp, b].conjugate()
    sys.tLbbp = tLbbp
    return 0

def generate_kern_lindblad_elph(sys):
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
                        for (l, q) in itertools.product(range(si.nbaths), range(2)):
                            fct_aap += tLbbp[l, b, a, q]*tLbbp[l, bp, ap, q].conjugate()
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
                            for (l, q) in itertools.product(range(si.nbaths), range(2)):
                                fct_bppbp += -0.5*tLbbp[l, a, b, q].conjugate()*tLbbp[l, a, bpp, q]
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
                            for (l, q) in itertools.product(range(si.nbaths), range(2)):
                                fct_bbpp += -0.5*tLbbp[l, a, bpp, q].conjugate()*tLbbp[l, a, bp, q]
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
    sys.kern = kern
    return 0

class Approach_pyLindblad(Approach_elph):

    kerntype = 'pyLindblad'
    generate_fct = staticmethod(generate_tLba)
    generate_kern = staticmethod(generate_kern_lindblad)
    generate_current = staticmethod(generate_current_lindblad)
    generate_vec = staticmethod(generate_vec_lindblad)
    #
    generate_kern_elph = staticmethod(generate_kern_lindblad_elph)
    generate_fct_elph = staticmethod(generate_tLbbp_elph)
#---------------------------------------------------------------------------------------------------------
